# -*- coding: utf-8 -*-
"""数据库连接配置（糖尿病风险预测平台）

本地免安装版 MySQL 8.4：
- root 账户仅用于初始化（空密码，仅限本机）
- 应用程序统一使用 diabetes_app 账户
"""
import os

# 项目根目录（scripts 的上一级）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 应用账户（平台运行时使用）
APP_DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "diabetes_app",
    "password": "Diabetes@2026",
    "database": "diabetes_platform",
    "charset": "utf8mb4",
}

# root 账户（仅 01_init_db.py 初始化建库建用户时使用）
ROOT_DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "charset": "utf8mb4",
    "local_infile": True,
}

DB_NAME = "diabetes_platform"

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

RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "pima-indians-diabetes.data.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "processed", "pima-indians-diabetes.clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
