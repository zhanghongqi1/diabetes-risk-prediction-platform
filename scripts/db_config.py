# -*- coding: utf-8 -*-
"""数据库连接配置（糖尿病风险预测平台）

配置优先级：环境变量 > 项目根目录 .env 文件 > 内置演示默认值。
内置默认值仅用于本机演示环境；生产或共享环境请务必用环境变量 / .env 覆盖
（.env 已加入 .gitignore，不会提交）。

可用环境变量：
    DIABETES_DB_HOST / DIABETES_DB_PORT / DIABETES_DB_NAME
    DIABETES_DB_USER / DIABETES_DB_PASSWORD            应用账号（平台运行时使用）
    DIABETES_DB_ROOT_USER / DIABETES_DB_ROOT_PASSWORD  root 账号（仅 01 初始化用）
"""
import os

# 项目根目录（scripts 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_dotenv(path: str) -> None:
    """极简 .env 解析（KEY=VALUE 行、# 注释），不引入第三方依赖。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(os.path.join(BASE_DIR, ".env"))


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v not in (None, "") else default


# 应用账户（平台运行时使用）
APP_DB_CONFIG = {
    "host": _env("DIABETES_DB_HOST", "localhost"),
    "port": int(_env("DIABETES_DB_PORT", "3306")),
    "user": _env("DIABETES_DB_USER", "diabetes_app"),
    "password": _env("DIABETES_DB_PASSWORD", "Diabetes@2026"),  # 演示默认值，仅限本机
    "database": _env("DIABETES_DB_NAME", "diabetes_platform"),
    "charset": "utf8mb4",
}

# root 账户（仅 01_init_db.py 初始化建库建用户时使用）
ROOT_DB_CONFIG = {
    "host": APP_DB_CONFIG["host"],
    "port": APP_DB_CONFIG["port"],
    "user": _env("DIABETES_DB_ROOT_USER", "root"),
    "password": _env("DIABETES_DB_ROOT_PASSWORD", ""),  # 本机绿色版 MySQL root 默认空密码
    "charset": "utf8mb4",
    "local_infile": True,
}

DB_NAME = APP_DB_CONFIG["database"]
# 01_init_db.py 用于同步 init_db.sql 中应用账号的密码
APP_DB_PASSWORD = APP_DB_CONFIG["password"]

# 8 项生理指标（与 UCI Pima 数据集一致）
FEATURE_COLUMNS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age",
]
FEATURE_LABELS_CN = {
    "pregnancies": "怀孕次数",
    "glucose": "血糖(糖耐量2h)",
    "blood_pressure": "舒张压",
    "skin_thickness": "皮褶厚度",
    "insulin": "血清胰岛素",
    "bmi": "BMI",
    "diabetes_pedigree_function": "糖尿病谱系功能",
    "age": "年龄",
}
# 医学上不可能为 0、0 实际代表缺失的指标
ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]

# 单例/批量输入的合法取值范围（临床上不可能的取值直接拒绝并提示；
# ZERO_AS_MISSING 字段允许填 0，表示"未测"，由模型 Pipeline 按训练集中位数自动填充）
INPUT_RANGES = {
    "pregnancies": (0, 20),
    "glucose": (40, 300),
    "blood_pressure": (30, 180),
    "skin_thickness": (5, 100),
    "insulin": (15, 900),
    "bmi": (10, 70),
    "diabetes_pedigree_function": (0.05, 2.5),
    "age": (18, 120),
}

RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "pima-indians-diabetes.data.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "pima-indians-diabetes.clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
