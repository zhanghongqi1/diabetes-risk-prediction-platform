# -*- coding: utf-8 -*-
"""02 数据导入与清洗

- 读取 UCI Pima 糖尿病数据集（768 条 × 8 项生理指标 + 标签）
- 原始数据写入 MySQL patient_raw 表
- 异常 0 值（Glucose/BloodPressure/SkinThickness/Insulin/BMI）按中位数填充
- 清洗结果写入 patient_clean 表，并导出带表头 CSV（data/processed/）

可重复执行（先 TRUNCATE 再导入）。
"""
import os
import sys
import numpy as np
import pandas as pd
import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import (APP_DB_CONFIG, RAW_CSV, CLEAN_CSV, FEATURE_COLUMNS,
                       ZERO_AS_MISSING)

COLUMNS = FEATURE_COLUMNS + ["outcome"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, header=None, names=COLUMNS)
    assert df.shape == (768, 9), f"原始数据维度异常: {df.shape}"
    return df


def clean(df_raw: pd.DataFrame):
    df = df_raw.copy()
    report = {}
    for col in ZERO_AS_MISSING:
        n_missing = int((df[col] == 0).sum())
        median = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, np.nan).fillna(median).astype(float)
        report[col] = (n_missing, round(float(median), 2))
    return df, report


def write_db(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    conn = pymysql.connect(**APP_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE patient_raw")
            cur.execute("TRUNCATE TABLE patient_clean")

            raw_rows = [
                (i + 1, int(r.pregnancies), int(r.glucose), int(r.blood_pressure),
                 int(r.skin_thickness), int(r.insulin), float(r.bmi),
                 float(r.diabetes_pedigree_function), int(r.age), int(r.outcome))
                for i, r in enumerate(df_raw.itertuples(index=False))
            ]
            cur.executemany(
                """INSERT INTO patient_raw
                   (id,pregnancies,glucose,blood_pressure,skin_thickness,insulin,
                    bmi,diabetes_pedigree_function,age,outcome)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", raw_rows)

            clean_rows = [
                (i + 1, float(r.pregnancies), float(r.glucose), float(r.blood_pressure),
                 float(r.skin_thickness), float(r.insulin), float(r.bmi),
                 float(r.diabetes_pedigree_function), float(r.age), int(r.outcome))
                for i, r in enumerate(df_clean.itertuples(index=False))
            ]
            cur.executemany(
                """INSERT INTO patient_clean
                   (id,pregnancies,glucose,blood_pressure,skin_thickness,insulin,
                    bmi,diabetes_pedigree_function,age,outcome)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", clean_rows)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM patient_raw")
            n_raw = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM patient_clean")
            n_clean = cur.fetchone()[0]
            cur.execute("SELECT outcome, COUNT(*) FROM patient_raw GROUP BY outcome")
            dist = cur.fetchall()
        return n_raw, n_clean, dist
    finally:
        conn.close()


def main():
    print("读取原始数据:", RAW_CSV)
    df_raw = load_raw()
    print("原始数据维度:", df_raw.shape, " 阳性/阴性:",
          int(df_raw.outcome.sum()), "/", int((df_raw.outcome == 0).sum()))

    df_clean, report = clean(df_raw)
    print("\n异常 0 值填充（中位数）:")
    for col, (n, med) in report.items():
        print(f"  {col:28s} 缺失 {n:3d} 处 -> 中位数 {med}")

    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
    print("\n清洗 CSV 已导出:", CLEAN_CSV)

    n_raw, n_clean, dist = write_db(df_raw, df_clean)
    print(f"\nMySQL 导入完成: patient_raw={n_raw} 行, patient_clean={n_clean} 行")
    print("标签分布:", dict(dist))
    assert n_raw == 768 and n_clean == 768
    print("数据导入验证通过。")


if __name__ == "__main__":
    main()
