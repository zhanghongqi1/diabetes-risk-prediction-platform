# -*- coding: utf-8 -*-
"""pytest 路径配置：使测试可直接 import web/app.py 与 scripts/db_config.py。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "web"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
