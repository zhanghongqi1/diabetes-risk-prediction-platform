# -*- coding: utf-8 -*-
"""01 初始化数据库：执行 init_db.sql（建库、建用户、建表）

用法：
    venv\\Scripts\\python.exe scripts\\01_init_db.py
可重复执行。
"""
import os
import sys
import pymysql

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import ROOT_DB_CONFIG, DB_NAME, APP_DB_PASSWORD

SQL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "init_db.sql")


def split_sql(sql_text: str):
    """按分号拆分 SQL 语句（本脚本无存储过程/触发器，简单拆分即可）。"""
    statements, buf = [], []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buf))
            buf = []
    if buf:
        statements.append("\n".join(buf))
    return [s.strip().rstrip(";").strip() for s in statements if s.strip()]


def main():
    with open(SQL_FILE, "r", encoding="utf-8") as f:
        sql_text = f.read()
    # 若通过环境变量 / .env 修改了应用账号密码，同步替换 SQL 中的默认密码
    if APP_DB_PASSWORD != "Diabetes@2026":
        sql_text = sql_text.replace(
            "IDENTIFIED BY 'Diabetes@2026'",
            "IDENTIFIED BY '%s'" % APP_DB_PASSWORD.replace("'", "''"))
    statements = split_sql(sql_text)

    conn = pymysql.connect(**ROOT_DB_CONFIG)
    try:
        with conn.cursor() as cur:
            for stmt in statements:
                head = stmt.split()[0].upper() if stmt.split() else ""
                # USE 语句 pymysql 需单独处理
                if stmt.upper().startswith("USE "):
                    conn.select_db(stmt.split(None, 1)[1].strip("`"))
                    print("[OK] 切换数据库")
                    continue
                cur.execute(stmt)
                print(f"[OK] {head} 语句执行成功")
        conn.commit()
    finally:
        conn.close()

    # 验证表
    conn = pymysql.connect(**{**ROOT_DB_CONFIG, "database": DB_NAME})
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [r[0] for r in cur.fetchall()]
        print("\n数据库 %s 中的表：%s" % (DB_NAME, ", ".join(tables)))
        expected = {"patient_raw", "patient_clean", "prediction_records", "model_info"}
        missing = expected - set(tables)
        if missing:
            print("[警告] 缺少表：", missing)
            sys.exit(1)
        print("数据库初始化验证通过。")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
