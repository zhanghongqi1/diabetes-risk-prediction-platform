# -*- coding: utf-8 -*-
"""03 模型训练、调参与选型

流程（与简历项目经历一致）：
- 从 MySQL patient_clean 读取清洗后数据
- 按 8:2 分层划分训练集/测试集（random_state=42，可复现）
- LR / SVM / GBDT 三类模型 Pipeline
- 五折交叉验证 + 网格搜索（GridSearchCV, scoring=roc_auc）
- 测试集对比 accuracy / precision / recall / F1 / AUC
- 模型 joblib 持久化到 models/，指标写入 model_info 表
- 输出特征相关性热力图、指标对比图、GBDT 特征贡献度
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import APP_DB_CONFIG, FEATURE_COLUMNS, FEATURE_LABELS_CN, MODEL_DIR

RANDOM_STATE = 42
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_from_mysql() -> pd.DataFrame:
    conn = pymysql.connect(**APP_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT " + ",".join(FEATURE_COLUMNS + ["outcome"]) +
                        " FROM patient_clean ORDER BY id")
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["outcome"])
    finally:
        conn.close()


def build_models():
    return {
        "LR": {
            "pipeline": Pipeline([("scaler", StandardScaler()),
                                  ("clf", LogisticRegression(max_iter=2000,
                                                             random_state=RANDOM_STATE))]),
            "params": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
            "file": "lr_pipeline.pkl",
        },
        "SVM": {
            # sklearn 1.9 起 SVC(probability=True) 弃用，改用 CalibratedClassifierCV
            "pipeline": Pipeline([("scaler", StandardScaler()),
                                  ("clf", CalibratedClassifierCV(
                                      SVC(kernel="rbf", random_state=RANDOM_STATE),
                                      cv=5, ensemble=False))]),
            "params": {"clf__estimator__C": [0.5, 1.0, 5.0],
                       "clf__estimator__gamma": ["scale", 0.1]},
            "file": "svm_pipeline.pkl",
        },
        "GBDT": {
            "pipeline": Pipeline([("clf", GradientBoostingClassifier(
                random_state=RANDOM_STATE))]),
            "params": {"clf__n_estimators": [100, 200],
                       "clf__max_depth": [2, 3],
                       "clf__learning_rate": [0.03, 0.05, 0.1],
                       "clf__min_samples_leaf": [5, 20],
                       "clf__subsample": [0.8, 1.0]},
            "file": "gbdt_pipeline.pkl",
        },
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }


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
    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS + ["outcome"]].corr()
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
    plt.title("LR / SVM / GBDT 测试集指标对比")
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
    print("从 MySQL 读取数据:", df.shape)
    X = df[FEATURE_COLUMNS]
    y = df["outcome"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"训练集 {X_train.shape[0]} 条 / 测试集 {X_test.shape[0]} 条（8:2 分层抽样）\n")

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
        m = evaluate(gs.best_estimator_, X_test, y_test)
        print(f"  最优参数: {gs.best_params_}")
        print(f"  CV-AUC = {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  测试集: ACC={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}\n")
        joblib.dump(gs.best_estimator_, os.path.join(MODEL_DIR, spec["file"]))
        fitted[name] = gs.best_estimator_
        rows.append({"model_name": name, **m,
                     "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
                     "model_path": f"models/{spec['file']}"})

    # 选型规则（与简历“对比精确率/召回率/F1/AUC 选型”一致）：
    # 1) 五折 CV-AUC 用于确认稳定性，与最优值差距在 1 个标准差内视为同档；
    # 2) 同档模型按测试集 AUC、F1、准确率综合排序，确定上线模型。
    best_cv = max(r["cv_auc_mean"] for r in rows)
    best_std = next(r["cv_auc_std"] for r in rows if r["cv_auc_mean"] == best_cv)
    candidates = [r for r in rows if best_cv - r["cv_auc_mean"] <= best_std]
    selected = max(candidates,
                   key=lambda r: (r["auc"], r["f1"], r["accuracy"]))["model_name"]
    print(f"\nCV-AUC 同档候选: {[r['model_name'] for r in candidates]} "
          f"(阈值 {best_cv:.4f}-{best_std:.4f})")
    joblib.dump(fitted[selected], os.path.join(MODEL_DIR, "selected_model.pkl"))

    # GBDT 特征贡献度
    gbdt = fitted["GBDT"]
    importances = gbdt.named_steps["clf"].feature_importances_
    fi = (pd.DataFrame({"feature": FEATURE_COLUMNS,
                        "feature_cn": [FEATURE_LABELS_CN[c] for c in FEATURE_COLUMNS],
                        "importance": importances})
          .sort_values("importance", ascending=False))
    fi.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"),
              index=False, encoding="utf-8-sig")
    meta = {"selected_model": selected,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": {r["model_name"]:
                        {k: round(v, 4) for k, v in r.items()
                         if k in ("accuracy", "precision", "recall", "f1", "auc",
                                  "cv_auc_mean", "cv_auc_std")}
                        for r in rows}}
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    plot_metrics(rows, os.path.join(img_dir, "model_metrics.png"))
    save_metrics(rows, selected)

    print("=" * 56)
    print(pd.DataFrame(rows)[["model_name", "accuracy", "precision", "recall",
                              "f1", "auc", "cv_auc_mean", "cv_auc_std"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 56)
    print(f"选型结果: {selected}（AUC 最高），已保存 models/selected_model.pkl")
    print("指标已写入 MySQL model_info 表，图表已输出到 web/static/img/")


if __name__ == "__main__":
    main()
