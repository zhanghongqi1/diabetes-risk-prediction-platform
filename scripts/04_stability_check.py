# -*- coding: utf-8 -*-
"""04 多种子稳定性验证

用 03 确定的 GBDT 最优超参（读自 models/model_meta.json），
在 5 个不同随机种子的 8:2 分层划分下重新训练并评估，
检验模型指标对数据划分是否敏感，给出 mean±std 稳定性结论。

用法：
    venv\\Scripts\\python.exe scripts\\04_stability_check.py
前置：已执行 02_import_data.py 与 03_train_models.py。
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import pymysql

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import APP_DB_CONFIG, FEATURE_COLUMNS, ZERO_AS_MISSING, MODEL_DIR

SEEDS = [0, 7, 21, 42, 99]


def load_raw() -> pd.DataFrame:
    conn = pymysql.connect(**APP_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT " + ",".join(FEATURE_COLUMNS + ["outcome"]) +
                        " FROM patient_raw ORDER BY id")
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["outcome"])
        return df.astype({c: float for c in FEATURE_COLUMNS})
    finally:
        conn.close()


def build_gbdt(best_params: dict) -> Pipeline:
    """按 03 的最优超参重建与训练一致的 Pipeline（含防泄漏填充）。"""
    kwargs = {k.replace("clf__", ""): v for k, v in best_params.items()}
    return Pipeline([
        ("prep", ColumnTransformer(
            [("zero2median", SimpleImputer(missing_values=0, strategy="median"),
              ZERO_AS_MISSING)],
            remainder="passthrough", verbose_feature_names_out=False)),
        ("clf", GradientBoostingClassifier(random_state=42, **kwargs)),
    ])


def main():
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    best_params = meta["metrics"]["GBDT"]["best_params"]
    threshold = float(meta["threshold"])
    print(f"GBDT 最优超参: {best_params}\n部署阈值 τ={threshold}\n")

    df = load_raw()
    X, y = df[FEATURE_COLUMNS], df["outcome"].astype(int)

    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed)
        pipe = build_gbdt(best_params)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        for tag, t in (("τ=0.5", 0.5), (f"τ={threshold}", threshold)):
            pred = (prob >= t).astype(int)
            rows.append({"seed": seed, "threshold": tag,
                         "accuracy": accuracy_score(y_test, pred),
                         "precision": precision_score(y_test, pred, zero_division=0),
                         "recall": recall_score(y_test, pred),
                         "f1": f1_score(y_test, pred),
                         "auc": roc_auc_score(y_test, prob)})

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = {}
    for tag, grp in df_out.groupby("threshold"):
        summary[tag] = {m: {"mean": round(float(grp[m].mean()), 4),
                            "std": round(float(grp[m].std()), 4),
                            "min": round(float(grp[m].min()), 4),
                            "max": round(float(grp[m].max()), 4)}
                        for m in ("accuracy", "precision", "recall", "f1", "auc")}

    print("\n===== 5 种子稳定性汇总（mean ± std）=====")
    for tag, metrics in summary.items():
        print(f"[{tag}] " + "  ".join(
            f"{m}={v['mean']:.4f}±{v['std']:.4f}" for m, v in metrics.items()))

    out = {"seeds": SEEDS, "best_params": best_params,
           "deploy_threshold": threshold, "detail": rows, "summary": summary}
    with open(os.path.join(MODEL_DIR, "stability.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已保存 models/stability.json")


if __name__ == "__main__":
    main()
