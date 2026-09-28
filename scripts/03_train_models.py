# -*- coding: utf-8 -*-
"""03 模型训练、调参、选型与筛查阈值优化

方法学要点（v2，修复数据泄漏 + 阈值优化）：
- 从 MySQL patient_raw 读取**原始**数据（异常 0 值保持原样入库）
- 先按 8:2 分层划分训练/测试集，再做任何依赖数据统计量的处理
- 缺失值填充（0 值 → 训练集中位数）通过 ColumnTransformer + SimpleImputer
  放进 sklearn Pipeline，仅在训练集（含交叉验证每个训练折）上拟合，
  从根源避免测试集信息泄漏；LR/SVM 的标准化同理置于 Pipeline 内
- 模型选型只依据五折交叉验证 CV-AUC，测试集不参与选型、仅作最终无偏报告；
  CV-AUC 差距在 1 个标准差内视为统计同档，同档时按部署需求选定 GBDT
  （产品端需要特征贡献度可解释性，且树模型能捕捉生理指标的非线性关系）
- 针对糖尿病筛查场景（漏诊代价高于误诊），用训练集 OOF 概率
  （cross_val_predict）做判定阈值调优：召回率 ≥ 0.70 前提下最大化精确率，
  阈值随模型写入 model_meta.json 一同部署
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pymysql
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import (APP_DB_CONFIG, FEATURE_COLUMNS, FEATURE_LABELS_CN,
                       ZERO_AS_MISSING, MODEL_DIR)

RANDOM_STATE = 42
# 筛查场景阈值调优目标：OOF 召回率不低于该值
TARGET_RECALL = 0.70
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_from_mysql() -> pd.DataFrame:
    """读取 patient_raw 原始数据（0 值不填充，交由 Pipeline 处理）。"""
    conn = pymysql.connect(**APP_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT " + ",".join(FEATURE_COLUMNS + ["outcome"]) +
                        " FROM patient_raw ORDER BY id")
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["outcome"])
        # DECIMAL 列经 pymysql 读出为 decimal.Decimal，统一转 float
        return df.astype({c: float for c in FEATURE_COLUMNS})
    finally:
        conn.close()


def build_preprocessor() -> ColumnTransformer:
    """0 值 → 训练集中位数（仅 5 项生理上不可能为 0 的指标），其余列原样通过。

    拟合发生在 Pipeline 内部，GridSearchCV / cross_val_predict 会自动保证
    每个训练折独立拟合填充值，测试折/测试集只做 transform。
    """
    return ColumnTransformer(
        [("zero2median", SimpleImputer(missing_values=0, strategy="median"),
          ZERO_AS_MISSING)],
        remainder="passthrough", verbose_feature_names_out=False)


def build_models():
    prep = build_preprocessor
    return {
        "LR": {
            "pipeline": Pipeline([("prep", prep()),
                                  ("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=2000,
                                                             random_state=RANDOM_STATE))]),
            "params": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            "file": "lr_pipeline.pkl",
        },
        "SVM": {
            # sklearn 1.9 起 SVC(probability=True) 弃用，改用 CalibratedClassifierCV
            "pipeline": Pipeline([("prep", prep()),
                                  ("scaler", StandardScaler()),
                                  ("clf", CalibratedClassifierCV(
                                      SVC(kernel="rbf", random_state=RANDOM_STATE),
                                      cv=5, ensemble=False))]),
            "params": {"clf__estimator__C": [0.5, 1.0, 5.0],
                       "clf__estimator__gamma": ["scale", 0.1]},
            "file": "svm_pipeline.pkl",
        },
        "GBDT": {
            "pipeline": Pipeline([("prep", prep()),
                                  ("clf", GradientBoostingClassifier(
                                      random_state=RANDOM_STATE))]),
            "params": {"clf__n_estimators": [100, 200],
                       "clf__max_depth": [2, 3],
                       "clf__learning_rate": [0.03, 0.05, 0.1],
                       "clf__min_samples_leaf": [5, 20],
                       "clf__subsample": [0.8, 1.0]},
            "file": "gbdt_pipeline.pkl",
        },
    }


def evaluate(model, X_test, y_test, threshold: float = 0.5):
    """threshold 用于判定类别；AUC 与阈值无关。"""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }


def tune_threshold(model, X_train, y_train, target_recall: float = TARGET_RECALL):
    """训练集 OOF 概率上调优判定阈值（不使用测试集）。

    规则：满足 OOF 召回率 >= target_recall 的阈值中，取精确率最高者；
    并列时取较大阈值（更稳健）。返回 (threshold, oof_precision, oof_recall)。
    """
    oof_prob = cross_val_predict(model, X_train, y_train, cv=5,
                                 method="predict_proba")[:, 1]
    best = None
    for t in np.arange(0.05, 0.951, 0.005):
        pred = (oof_prob >= t).astype(int)
        r = recall_score(y_train, pred)
        p = precision_score(y_train, pred, zero_division=0)
        if r >= target_recall and (best is None or (p, t) > best[0]):
            best = ((p, t), t, p, r)
    if best is None:  # 极端情况：目标召回不可达，退回 0.5
        return 0.5, None, None
    _, t, p, r = best
    return round(float(t), 3), round(float(p), 4), round(float(r), 4)


def export_gbdt_json(pipe: Pipeline, path: str):
    """导出 GBDT 纯 Python 推理参数（供无 sklearn 环境的部署端使用，如扣子智能体）。

    推理方式：
      1) 对 ZERO_AS_MISSING 中的字段，0 值替换为 impute_medians 中的训练集中位数；
      2) 按 feature_names 顺序组装特征向量，逐棵树行走累加叶子值；
      3) score = f0 + learning_rate * Σ叶子值，prob = 1 / (1 + exp(-score))。
    """
    ct = pipe.named_steps["prep"]
    gb: GradientBoostingClassifier = pipe.named_steps["clf"]
    feature_names = list(ct.get_feature_names_out())
    imputer = ct.named_transformers_["zero2median"]
    medians = {c: float(v) for c, v in zip(ZERO_AS_MISSING, imputer.statistics_)}

    p1 = float(gb.init_.class_prior_[1])
    f0 = float(np.log(p1 / (1 - p1)))

    def node_to_dict(tree, n):
        if int(tree.children_left[n]) == -1:  # 叶子
            return {"feature": -1, "value": float(np.ravel(tree.value[n])[0])}
        return {"feature": int(tree.feature[n]),
                "threshold": float(tree.threshold[n]),
                "left": node_to_dict(tree, int(tree.children_left[n])),
                "right": node_to_dict(tree, int(tree.children_right[n]))}

    trees = [node_to_dict(est.tree_, 0) for est in gb.estimators_[:, 0]]
    payload = {
        "format": "sklearn.GradientBoostingClassifier",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "zero_as_missing": ZERO_AS_MISSING,
        "impute_medians": medians,
        "inference": "对 zero_as_missing 字段将 0 替换为 impute_medians；"
                     "score = f0 + learning_rate * Σ叶子值；prob = sigmoid(score)",
        "f0": f0,
        "learning_rate": float(gb.learning_rate),
        "n_trees": len(trees),
        "trees": trees,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


def verify_gbdt_json(payload: dict, pipe: Pipeline, X_sample: pd.DataFrame):
    """纯 Python 推理 vs sklearn Pipeline 推理一致性校验（含 0 值填充）。"""
    names = payload["feature_names"]
    meds = payload["impute_medians"]

    def walk(node, x):
        while node["feature"] != -1:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def py_prob(row):
        x = [float(row[c]) for c in names]
        x = [meds[c] if (c in meds and v == 0) else v for c, v in zip(names, x)]
        s = payload["f0"] + payload["learning_rate"] * sum(walk(t, x)
                                                           for t in payload["trees"])
        return 1 / (1 + np.exp(-s))

    sk = pipe.predict_proba(X_sample)[:, 1]
    py = np.array([py_prob(row) for _, row in X_sample.iterrows()])
    return float(np.max(np.abs(sk - py)))


def save_metrics(rows, selected):
    conn = pymysql.connect(**APP_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE model_info")
            for r in rows:
                cur.execute(
                    """INSERT INTO model_info
                       (model_name,accuracy,precision_score,recall_score,f1_score,auc,
                        cv_auc_mean,cv_auc_std,model_path,is_selected)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (r["model_name"], r["accuracy"], r["precision"], r["recall"],
                     r["f1"], r["auc"], r["cv_auc_mean"], r["cv_auc_std"],
                     r["model_path"], 1 if r["model_name"] == selected else 0))
        conn.commit()
    finally:
        conn.close()


def plot_corr(df, out_path):
    """相关性热力图（描述性统计展示用：0 值按全量中位数填充，仅用于绘图）。"""
    d = df.copy()
    for col in ZERO_AS_MISSING:
        d[col] = d[col].replace(0, np.nan).fillna(d.loc[d[col] != 0, col].median())
    plt.figure(figsize=(9, 7))
    corr = d[FEATURE_COLUMNS + ["outcome"]].corr()
    labels = [FEATURE_LABELS_CN[c] for c in FEATURE_COLUMNS] + ["糖尿病"]
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=labels, yticklabels=labels, square=True, linewidths=.5)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metrics(rows, out_path):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    names_cn = ["准确率", "精确率", "召回率", "F1", "AUC"]
    models = [r["model_name"] for r in rows]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(10, 5.5))
    for i, m in enumerate(models):
        vals = [next(r for r in rows if r["model_name"] == m)[k] for k in metrics]
        plt.bar(x + (i - 1) * width, vals, width, label=m)
    plt.xticks(x, names_cn)
    plt.ylim(0.5, 1.0)
    plt.ylabel("分数")
    plt.title("LR / SVM / GBDT 测试集指标对比（判定阈值 0.5）")
    plt.legend()
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    img_dir = os.path.join(os.path.dirname(MODEL_DIR), "web", "static", "img")
    os.makedirs(img_dir, exist_ok=True)

    df = load_from_mysql()
    print("从 MySQL patient_raw 读取原始数据:", df.shape)
    X = df[FEATURE_COLUMNS]
    y = df["outcome"].astype(int)

    # 先划分，后训练：所有统计量（填充中位数/标准化）仅在训练集上拟合
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"训练集 {X_train.shape[0]} 条 / 测试集 {X_test.shape[0]} 条（8:2 分层抽样）")
    print("缺失值填充与标准化已置于 Pipeline 内，仅用训练集统计量（防泄漏）\n")

    plot_corr(df, os.path.join(img_dir, "corr_heatmap.png"))

    rows = []
    fitted = {}
    for name, spec in build_models().items():
        print(f"===== {name}：五折交叉验证 + 网格搜索 =====")
        # 说明：项目路径含中文，并行工作进程的临时目录可能触发非 ASCII
        # 编码兼容问题；数据集仅 768 条，串行网格搜索秒级完成，故 n_jobs=1
        gs = GridSearchCV(spec["pipeline"], spec["params"], cv=5,
                          scoring="roc_auc", n_jobs=1)
        gs.fit(X_train, y_train)
        cv_mean = float(gs.best_score_)
        cv_std = float(gs.cv_results_["std_test_score"][gs.best_index_])
        m = evaluate(gs.best_estimator_, X_test, y_test, threshold=0.5)
        print(f"  最优参数: {gs.best_params_}")
        print(f"  CV-AUC = {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  测试集(阈值0.5): ACC={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}\n")
        joblib.dump(gs.best_estimator_, os.path.join(MODEL_DIR, spec["file"]))
        fitted[name] = gs.best_estimator_
        rows.append({"model_name": name, **m,
                     "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
                     "best_params": gs.best_params_,
                     "model_path": f"models/{spec['file']}"})

    # 选型规则（v2：只使用交叉验证指标，测试集不参与选型）：
    # 1) 五折 CV-AUC 与最优值差距在 1 个标准差内视为统计同档；
    # 2) 同档时按部署需求选定 GBDT——产品端需要特征贡献度支撑可解释性展示，
    #    且树模型能捕捉生理指标的非线性关系；若 GBDT 不在同档内则取 CV-AUC 最高者。
    best_cv = max(r["cv_auc_mean"] for r in rows)
    best_std = next(r["cv_auc_std"] for r in rows if r["cv_auc_mean"] == best_cv)
    candidates = [r for r in rows if best_cv - r["cv_auc_mean"] <= best_std]
    cand_names = [r["model_name"] for r in candidates]
    if "GBDT" in cand_names:
        selected = "GBDT"
        reason = ("CV-AUC 统计同档（差距<1标准差），按部署需求选 GBDT："
                  "特征重要性支撑产品端贡献度展示，树模型可捕捉非线性关系")
    else:
        selected = max(candidates, key=lambda r: r["cv_auc_mean"])["model_name"]
        reason = "GBDT 未进入 CV-AUC 同档，取 CV-AUC 最高者"
    print(f"CV-AUC 同档候选: {cand_names} (最优 {best_cv:.4f}±{best_std:.4f})")
    print(f"选型结果: {selected} —— {reason}\n")
    joblib.dump(fitted[selected], os.path.join(MODEL_DIR, "selected_model.pkl"))

    # 筛查阈值调优（仅训练集 OOF，不碰测试集）
    threshold, oof_p, oof_r = tune_threshold(fitted[selected], X_train, y_train)
    print(f"判定阈值调优（目标 OOF 召回率≥{TARGET_RECALL:.2f}）: τ={threshold} "
          f"（OOF 精确率={oof_p} 召回率={oof_r}）")
    m_sel = evaluate(fitted[selected], X_test, y_test, threshold=threshold)
    print(f"部署阈值下测试集: ACC={m_sel['accuracy']:.4f} P={m_sel['precision']:.4f} "
          f"R={m_sel['recall']:.4f} F1={m_sel['f1']:.4f} AUC={m_sel['auc']:.4f}\n")

    # GBDT 特征贡献度（列名经 ColumnTransformer 后仍为一一对应）
    gbdt = fitted["GBDT"]
    out_names = list(gbdt.named_steps["prep"].get_feature_names_out())
    importances = gbdt.named_steps["clf"].feature_importances_
    fi = (pd.DataFrame({"feature": out_names,
                        "feature_cn": [FEATURE_LABELS_CN[c] for c in out_names],
                        "importance": importances})
          .sort_values("importance", ascending=False))
    fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"),
              index=False, encoding="utf-8-sig")

    # 导出 GBDT 纯 Python 推理参数并做一致性校验
    payload = export_gbdt_json(gbdt, os.path.join(MODEL_DIR, "gbdt_trees.json"))
    diff = verify_gbdt_json(payload, gbdt, X_test.head(50))
    print(f"gbdt_trees.json 纯 Python 推理一致性校验: 最大误差 {diff:.2e}")

    def _py(v):
        return v.item() if hasattr(v, "item") else v

    meta = {
        "selected_model": selected,
        "selection_rule": reason,
        "threshold": threshold,
        "threshold_rule": f"训练集五折 OOF 概率，召回率≥{TARGET_RECALL:.2f} 时最大化精确率",
        "threshold_oof": {"precision": oof_p, "recall": oof_r},
        "deployed_test_metrics": {k: round(float(v), 4)
                                  for k, v in m_sel.items()},
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {r["model_name"]:
                    {k: round(_py(v), 4) if isinstance(_py(v), float) else _py(v)
                     for k, v in r.items()
                     if k in ("accuracy", "precision", "recall", "f1", "auc",
                              "cv_auc_mean", "cv_auc_std", "best_params")}
                    for r in rows},
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2,
                  default=lambda o: _py(o) if hasattr(o, "item") else str(o))

    plot_metrics(rows, os.path.join(img_dir, "model_metrics.png"))
    save_metrics(rows, selected)

    print("=" * 64)
    print(pd.DataFrame(rows)[["model_name", "accuracy", "precision", "recall",
                              "f1", "auc", "cv_auc_mean", "cv_auc_std"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 64)
    print(f"选型结果: {selected}；部署阈值 τ={threshold}（OOF R={oof_r}）")
    print("指标已写入 MySQL model_info 表，图表已输出到 web/static/img/")


if __name__ == "__main__":
    main()
