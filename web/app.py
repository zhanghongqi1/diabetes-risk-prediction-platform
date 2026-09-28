# -*- coding: utf-8 -*-
"""糖尿病风险预测平台 - Flask 后端

启动：
    venv\\Scripts\\python.exe web\\app.py
访问：
    http://127.0.0.1:5000

接口：
- GET  /                       前端页面
- GET  /api/health             健康检查（模型 + 数据库）
- GET  /api/meta               模型指标 / 特征重要性
- POST /api/predict            单例风险评估
- POST /api/predict/batch      批量导入（CSV/Excel）
- GET  /api/history            最近预测记录
- GET  /api/template           批量导入模板下载
"""
import os
import sys
import uuid
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib
import pymysql
from flask import Flask, request, jsonify, send_from_directory, send_file
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "scripts"))
from db_config import APP_DB_CONFIG, FEATURE_COLUMNS, FEATURE_LABELS_CN, MODEL_DIR

app = Flask(__name__, static_folder="static", static_url_path="/static")

model = joblib.load(os.path.join(MODEL_DIR, "selected_model.pkl"))
with open(os.path.join(MODEL_DIR, "model_meta.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
SELECTED = meta["selected_model"]
fi = pd.read_csv(os.path.join(MODEL_DIR, "feature_importance.csv"))


def db_conn():
    return pymysql.connect(**APP_DB_CONFIG)


def risk_level(p: float) -> str:
    if p < 1 / 3:
        return "低风险"
    if p < 2 / 3:
        return "中风险"
    return "高风险"


def predict_one(values: dict):
    x = np.array([[float(values[c]) for c in FEATURE_COLUMNS]])
    prob = float(model.predict_proba(x)[0, 1])
    cls = int(prob >= 0.5)
    return prob, cls, risk_level(prob)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/health")
def health():
    db_ok, db_err, n_clean = False, "", 0
    try:
        conn = db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM patient_clean")
            n_clean = cur.fetchone()[0]
        conn.close()
        db_ok = True
    except Exception as e:  # noqa
        db_err = str(e)
    return jsonify({"model_loaded": model is not None,
                    "selected_model": SELECTED,
                    "mysql_ok": db_ok, "mysql_error": db_err,
                    "patient_clean_rows": n_clean})


@app.route("/api/meta")
def get_meta():
    return jsonify({
        "selected_model": SELECTED,
        "metrics": meta["metrics"],
        "feature_columns": FEATURE_COLUMNS,
        "feature_labels_cn": FEATURE_LABELS_CN,
        "feature_importance": fi.to_dict(orient="records"),
    })


def _save_records(records, batch_no):
    cols = FEATURE_COLUMNS
    conn = db_conn()
    try:
        with conn.cursor() as cur:
            for r in records:
                cur.execute(
                    """INSERT INTO prediction_records
                       (batch_no,""" + ",".join(cols) + """,actual_outcome,
                        risk_probability,risk_level,predicted_class,model_name)
                       VALUES (%s,""" + ",".join(["%s"] * len(cols)) +
                    """,%s,%s,%s,%s,%s)""",
                    (batch_no, *[r.get(c) for c in cols], r.get("actual_outcome"),
                     r["risk_probability"], r["risk_level"], r["predicted_class"],
                     SELECTED))
        conn.commit()
    finally:
        conn.close()


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        prob, cls, level = predict_one(data)
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"输入参数有误: {e}"}), 400
    rec = {c: float(data[c]) for c in FEATURE_COLUMNS}
    rec.update(risk_probability=round(prob, 4), risk_level=level,
               predicted_class=cls, actual_outcome=data.get("actual_outcome"))
    _save_records([rec], batch_no=None)
    return jsonify({"risk_probability": round(prob, 4),
                    "risk_level": level,
                    "predicted_class": cls,
                    "model_name": SELECTED})


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "未收到文件"}), 400
    try:
        if f.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            return jsonify({"error": f"缺少列: {','.join(missing)}"}), 400
        X = df[FEATURE_COLUMNS].astype(float)
        probs = model.predict_proba(X)[:, 1]
        batch_no = dt.datetime.now().strftime("B%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
        records, results = [], []
        for i, (_, row) in enumerate(df.iterrows()):
            p = float(probs[i])
            cls = int(p >= 0.5)
            rec = {c: float(row[c]) for c in FEATURE_COLUMNS}
            rec["actual_outcome"] = (int(row["outcome"])
                                     if "outcome" in df.columns and pd.notna(row.get("outcome"))
                                     else None)
            rec.update(risk_probability=round(p, 4), risk_level=risk_level(p),
                       predicted_class=cls)
            records.append(rec)
            results.append({"row": i + 2, "risk_probability": round(p, 4),
                            "risk_level": risk_level(p), "predicted_class": cls})
        _save_records(records, batch_no)
        df_out = pd.DataFrame(results)
        summary = df_out["risk_level"].value_counts().to_dict()
        return jsonify({"batch_no": batch_no, "total": len(results),
                        "summary": summary, "results": results[:200]})
    except Exception as e:  # noqa
        return jsonify({"error": f"批量预测失败: {e}"}), 400


@app.route("/api/history")
def history():
    conn = db_conn()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cur:
            cur.execute("""SELECT id,batch_no,risk_probability,risk_level,
                                  predicted_class,model_name,created_at
                           FROM prediction_records ORDER BY id DESC LIMIT 20""")
            rows = cur.fetchall()
        for r in rows:
            r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            r["risk_probability"] = float(r["risk_probability"])
        return jsonify(rows)
    finally:
        conn.close()


@app.route("/api/template")
def template():
    df = pd.DataFrame(columns=FEATURE_COLUMNS + ["outcome(可选)"])
    buf = io.StringIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    buf.write("6,148,72,35,0,33.6,0.627,50,1\n")
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode("utf-8-sig")),
                     mimetype="text/csv",
                     as_attachment=True, download_name="batch_template.csv")


if __name__ == "__main__":
    print(f"选定模型: {SELECTED}，启动 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
