# -*- coding: utf-8 -*-
"""核心逻辑单元测试（不依赖 MySQL 连接）。

运行：
    venv\\Scripts\\python.exe -m pytest tests/ -v
"""
import pytest

import app as web  # web/app.py（加载 models/ 下的模型与 meta）

# README 示例样本（临床上合理的合法输入）
SAMPLE = {
    "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
    "skin_thickness": 35, "insulin": 125, "bmi": 33.6,
    "diabetes_pedigree_function": 0.627, "age": 50,
}


# ---------- 风险分层 ----------

def test_risk_level_boundaries():
    assert web.risk_level(0.0) == "低风险"
    assert web.risk_level(1 / 3 - 1e-9) == "低风险"
    assert web.risk_level(1 / 3) == "中风险"
    assert web.risk_level(2 / 3 - 1e-9) == "中风险"
    assert web.risk_level(2 / 3) == "高风险"
    assert web.risk_level(1.0) == "高风险"


# ---------- 输入校验 ----------

def test_validate_input_ok():
    out = web.validate_input(SAMPLE)
    assert out["glucose"] == 148.0
    assert set(out.keys()) == set(SAMPLE.keys())


def test_validate_input_zero_as_missing_allowed():
    """5 项生理上不可能为 0 的指标允许填 0（表示未测，交由模型填充）。"""
    for col in ("glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"):
        s = dict(SAMPLE)
        s[col] = 0
        assert web.validate_input(s)[col] == 0.0


def test_validate_input_out_of_range_rejected():
    for col, bad in (("glucose", -5), ("glucose", 500),
                     ("blood_pressure", 10), ("bmi", 80), ("age", 10),
                     ("pregnancies", 25)):
        s = dict(SAMPLE)
        s[col] = bad
        with pytest.raises(ValueError):
            web.validate_input(s)


def test_validate_input_missing_or_non_numeric():
    s = dict(SAMPLE)
    del s["bmi"]
    with pytest.raises(ValueError):
        web.validate_input(s)
    s = dict(SAMPLE)
    s["glucose"] = "abc"
    with pytest.raises(ValueError):
        web.validate_input(s)


# ---------- 单例预测 ----------

def test_predict_one_smoke():
    prob, cls, level = web.predict_one(web.validate_input(SAMPLE))
    assert 0.0 < prob < 1.0
    assert cls == int(prob >= web.THRESHOLD)  # 判定必须使用部署阈值
    assert level in ("低风险", "中风险", "高风险")


# ---------- Flask 接口（仅不依赖 MySQL 的路径） ----------

def test_template_endpoint():
    client = web.app.test_client()
    r = client.get("/api/template")
    assert r.status_code == 200
    assert "pregnancies" in r.data.decode("utf-8-sig")


def test_predict_endpoint_rejects_bad_input_without_db():
    """非法输入应在写库前被拦截，直接返回 400。"""
    client = web.app.test_client()
    r = client.post("/api/predict", json={**SAMPLE, "glucose": -5})
    assert r.status_code == 400
    assert "error" in r.get_json()
