# 基于机器学习的糖尿病风险预测与健康数据分析平台

基于 UCI Pima Indians Diabetes 数据集（768 条、8 项生理指标）的糖尿病风险预测平台：
Pandas/NumPy 完成数据探查与入库，8:2 分层划分训练/测试集；缺失值填充与标准化
封装在 Scikit-learn Pipeline 内**仅用训练集统计量拟合**（防数据泄漏），
搭建 **LR / SVM / GBDT** 三类模型，五折交叉验证 + 网格搜索调参，按 CV-AUC 选型上线；
针对筛查场景基于训练集 OOF 概率完成**判定阈值调优**（召回率 51.9% → 68.5%）；
Flask + 原生 HTML/JavaScript 前端提供风险仪表盘、特征贡献度图表，
支持单例查询与 CSV/Excel 批量导入，预测记录全部入库留痕。

**技术栈**：Python 3.13 · Pandas · NumPy · Scikit-learn · MySQL 8.4 · Flask · HTML/JavaScript · ECharts · joblib · pytest

> 本项目为课程设计（2025.10–2026.03，医学信息工程 3 人小组）结项后整理开源：
> 课程期间完成数据、模型与文档，开源前做了工程化重构与方法学修正（见「迭代记录」）。
> 小组分工：本人负责模型训练、训练/评估代码与 Web 应用开发；
> 另两位组员负责数据收集整理与课程报告撰写。

## 🖼️ 界面预览

![平台整页预览](docs/preview_full.png)

## ✨ 功能一览

| 模块 | 说明 |
|---|---|
| 单例风险评估 | 输入 8 项生理指标 → 风险概率 + 低/中/高风险仪表盘（含输入范围校验，5 项指标支持"未测填 0"自动填充） |
| 批量导入 | CSV / Excel 上传批量预测，逐行数据校验并定位非法行，提供模板下载，结果汇总+明细 |
| 筛查阈值优化 | 基于训练集 OOF 概率调优判定阈值（0.5 → 0.375），漏诊率显著下降 |
| GBDT 特征贡献度 | ECharts 柱状图展示 8 特征重要性排序 |
| 相关性热力图 | 训练阶段生成（matplotlib + seaborn），页面直读 |
| 模型指标对比 | LR / SVM / GBDT 准确率、精确率、召回率、F1、AUC、CV-AUC 对比 |
| 预测记录入库 | 单例与批量预测全部写入 MySQL `prediction_records` 表 |
| 单元测试 | pytest 覆盖风险分层、输入校验、单例预测与接口冒烟（见 `tests/`） |

## 📊 复现实验结果（测试集 154 例）

数据划分：8:2 分层抽样，`random_state=42`（完全可复现）。
防泄漏：Glucose/BloodPressure/SkinThickness/Insulin/BMI 的异常 0 值视为缺失，
由 Pipeline 内 `SimpleImputer` **仅用训练集中位数**填充（训练折内独立拟合）。
模型 Pipeline：LR（填充+标准化+网格调 C）、SVM（填充+标准化+概率校准+网格调 C/γ）、
GBDT（填充+网格调学习率/深度/树数/叶样本数/行采样共 48 组），均五折交叉验证（评分 roc_auc）。
选型规则（v2）：**仅依据五折 CV-AUC 选型，测试集不参与**；CV-AUC 差距在 1 个标准差内
视为统计同档，同档时按部署需求选定 GBDT（产品端需要特征重要性支撑贡献度展示，
且树模型能捕捉生理指标非线性关系）。

### 测试集指标（判定阈值 0.5，用于模型间公平对比）

| 模型 | 准确率 | 精确率 | 召回率 | F1 | AUC | 五折 CV-AUC |
|---|---|---|---|---|---|---|
| LR | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 | 0.8430±0.0291 |
| SVM | 0.7208 | 0.6222 | 0.5185 | 0.5657 | 0.8059 | 0.8336±0.0211 |
| **GBDT ★ 选定** | **0.7338** | **0.6512** | 0.5185 | **0.5773** | **0.8204** | **0.8440±0.0228** |

### 筛查阈值优化（部署口径，τ=0.375）

糖尿病筛查场景漏诊代价远高于误诊，默认 0.5 阈值下召回率仅约 0.52。
在**训练集五折 OOF 概率**（cross_val_predict，不碰测试集）上按
"召回率 ≥ 0.70 时最大化精确率"调优，得 τ=0.375（OOF 精确率 0.658 / 召回率 0.710）：

| 口径 | 准确率 | 精确率 | 召回率 | F1 | AUC |
|---|---|---|---|---|---|
| 测试集 @0.5 | 0.7338 | 0.6512 | 0.5185 | 0.5773 | 0.8204 |
| **测试集 @τ=0.375（部署）** | 0.7273 | 0.5968 | **0.6852** | **0.6379** | 0.8204 |

召回率 **+16.7 个百分点**、F1 **+6.1 个百分点**，准确率仅 -0.65 个百分点。

### 多种子稳定性（5 个随机种子复训，见 `scripts/04_stability_check.py`）

| 口径 | 准确率 | 召回率 | F1 | AUC |
|---|---|---|---|---|
| 5 种子 @0.5 | 0.7584±0.0277 | 0.5370±0.0797 | 0.6069±0.0632 | 0.8430±0.0177 |
| 5 种子 @τ=0.375 | 0.7623±0.0218 | **0.7111±0.0712** | **0.6763±0.0353** | 0.8430±0.0177 |

阈值调优的召回收益在不同数据划分下稳定存在（准确率不受损甚至略升）。
测试集仅 154 例，1 例 ≈ 0.65 个百分点，单种子指标存在 ±2~3 个百分点波动属正常。

## 🔁 迭代记录（v1 → v2，2026-09-28）

开源整理时对方法学与工程做了以下修正/增强，全部可复现：

1. **修复数据泄漏**：v1 在全量 768 条上做中位数填充后再划分；v2 改为
   `ColumnTransformer + SimpleImputer` 置于 Pipeline 内，仅用训练集（含 CV 训练折）统计量。
   修复后测试集指标与 v1 基本一致（该数据集上泄漏影响很小），但流程严格正确。
2. **选型不再使用测试集**：v1 同档内按测试集 AUC 选型（test peeking）；
   v2 仅依据 CV-AUC，同档按部署需求选 GBDT，测试集只作最终无偏报告。
3. **新增筛查阈值调优**：见上节，召回率 51.9% → 68.5%（测试集）。
4. **新增多种子稳定性验证**：`04_stability_check.py` + `models/stability.json`。
5. **工程化**：数据库密码支持环境变量 / `.env` 覆盖（`.env.example`，不再只有硬编码）；
   单例/批量接口新增输入范围校验（非法行定位到行号）；新增 pytest 单元测试（8 项）；
   补充 MIT LICENSE。
6. **推理一致性**：v2 的 `gbdt_trees.json` 增加 `impute_medians`，
   纯 Python 推理前先按训练集中位数填充 0 值，与 sklearn Pipeline 最大误差 3.3e-16。

> 指标口径说明：本 README 所有数字均为固定随机种子/多种子下的可复现实验结果，
> 与简历记载一致。v1 历史版本（简历曾记载 82%/0.85）属于特定数据划分与预处理
> 组合下的早期结果，已统一为当前可复现口径。

## 🚀 从零复现指南（新环境）

无需任何预置文件，按以下步骤可在全新 Windows 机器上完整跑通。

### 1. 环境准备

- **Python 3.13**：从 [python.org](https://www.python.org/downloads/) 安装，勾选 *Add python.exe to PATH*。
- **MySQL 8.4**：从 [dev.mysql.com](https://dev.mysql.com/downloads/installer/) 安装（或使用已有 MySQL 8.x 实例），默认端口 3306，保证服务已启动。

### 2. 获取代码并安装依赖

```powershell
git clone https://github.com/zhanghongqi1/diabetes-risk-prediction-platform.git
cd diabetes-risk-prediction-platform
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如需运行单元测试，改装 requirements-dev.txt
```

### 3. 配置数据库连接

默认配置（localhost:3306、root 空密码、应用账号 `diabetes_app`/`Diabetes@2026`）可直接跑通本机演示。
如需修改，复制 `.env.example` 为 `.env` 并编辑（`.env` 已被 .gitignore 忽略），
或直接设置同名环境变量；优先级：**环境变量 > .env > 内置演示默认值**。
root 密码仅 `01_init_db.py` 初始化时使用；应用账号由 01 脚本自动创建（密码与 `.env` 同步）。

### 4. 建库 → 导入数据 → 训练模型 → 稳定性验证（按顺序执行）

```powershell
.\venv\Scripts\python.exe scripts\01_init_db.py      # ① 建库建表 + 创建应用账号（幂等）
.\venv\Scripts\python.exe scripts\02_import_data.py  # ② 导入 768 条原始数据入库
.\venv\Scripts\python.exe scripts\03_train_models.py # ③ LR/SVM/GBDT 五折CV+网格搜索+选型+阈值调优+出图
.\venv\Scripts\python.exe scripts\04_stability_check.py # ④ 5 种子稳定性验证（可选）
```

### 5. 运行单元测试（可选）

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
.\venv\Scripts\python.exe -m pytest tests/ -v     # 8 项：风险分层/输入校验/预测冒烟/接口
```

### 6. 启动 Web 平台

```powershell
.\venv\Scripts\python.exe web\app.py
# 浏览器访问 http://127.0.0.1:5000
```

> 提示：端口 3306 被占用时，在 `.env` 中修改 `DIABETES_DB_PORT` 即可；
> Linux/macOS 用户将路径分隔符换为 `/`，其余步骤一致。

## 🗄️ 数据库说明

| 项 | 值 |
|---|---|
| 主机/端口 | localhost : 3306（可用 `.env` / 环境变量覆盖） |
| 数据库 | `diabetes_platform`（字符集 utf8mb4） |
| 应用账号 | `diabetes_app` / 默认 `Diabetes@2026`（由 01 脚本创建，仅本机演示环境，请勿用于生产；可用环境变量覆盖） |
| 表 | `patient_raw`（768 条原始）、`patient_clean`（768 条清洗示例，仅展示用）、`prediction_records`（预测记录）、`model_info`（模型指标） |

## 📁 目录结构

```
├── README.md               项目说明
├── LICENSE                 MIT
├── requirements.txt        运行依赖（锁定版本另见 requirements-lock.txt）
├── requirements-dev.txt    开发/测试依赖（含 pytest）
├── .env.example            环境变量模板（复制为 .env 使用，.env 不入库）
├── .gitignore
├── data/
│   ├── raw/                UCI 原始数据（768 行无表头 CSV + 字段说明 .names）
│   └── processed/          清洗后带表头 CSV（展示用）
├── models/                 训练产物：3 个模型 pkl、selected_model.pkl、特征重要性、
│                           指标 model_meta.json（含部署阈值）、stability.json、gbdt_trees.json
├── scripts/
│   ├── init_db.sql         建库/建用户/建表 SQL
│   ├── db_config.py        数据库连接（环境变量/.env/默认值三级配置）与字段、校验范围配置
│   ├── 01_init_db.py       ① 初始化数据库
│   ├── 02_import_data.py   ② 数据导入（原始入库 + 清洗示例）
│   ├── 03_train_models.py  ③ 训练 LR/SVM/GBDT：Pipeline 内填充（防泄漏）+
│   │                        五折CV+网格搜索 + 纯CV选型 + OOF 阈值调优 + 导出/校验
│   ├── 04_stability_check.py ④ 多种子稳定性验证
│   └── start_mysql.ps1 / stop_mysql.ps1   本机绿色版 MySQL 启停（克隆用户可忽略）
├── tests/                  pytest 单元测试（风险分层/输入校验/预测/接口冒烟）
├── web/
│   ├── app.py              Flask 后端（预测/批量/历史/模板接口，含输入校验与部署阈值）
│   ├── index.html          前端单页（仪表盘、表单、批量导入、指标、热力图）
│   └── static/             ECharts 本地库 + 生成的图表
├── docs/                   验收截图与运行日志
└── 启动平台.ps1            本机绿色环境一键启动（依赖本机 venv/mysql，克隆用户可忽略）
```

### Flask API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（模型 + 数据库 + 部署阈值） |
| GET | `/api/meta` | 模型指标 / 特征重要性 / 部署阈值及 OOF 依据 |
| POST | `/api/predict` | 单例风险评估（输入范围校验，0=未测自动填充） |
| POST | `/api/predict/batch` | 批量导入（CSV/Excel，逐行校验并定位非法行号） |
| GET | `/api/history` | 最近 20 条预测记录 |
| GET | `/api/template` | 批量导入 CSV 模板下载 |

## 🌐 在线演示

扣子智能体「糖尿病风险预测助手」已开发完成并部署上线（部署版本 9527b027）：

- **智能体在线体验（免登录）**：<https://code.coze.cn/web-sdk/7690389329864196096>（官方 Web SDK 公开分享链接，任何访客打开网页即可直接与智能体对话，无需登录、无需配置）
- **生产部署域名（API 服务）**：<https://8vzpst8fw8.coze.site>（Agent API 端点，按平台策略要求 Bearer Token 鉴权，适合集成方通过 OpenAPI 方式调用）
- **智能体项目页**：<https://code.coze.cn/p/7690389329864196096>（需登录扣子账号，供项目所有者/协作成员进入开发环境）

> 说明：线上智能体部署的是 v1 模型参数。仓库模型已迭代至 v2（见「迭代记录」），
> 重新部署时用最新 `models/gbdt_trees.json` 替换智能体参数文件即可；
> v2 参数新增 `impute_medians`，推理前会先按训练集中位数填充 0 值（医学上 0=未测）。

### 实测对话记录（2026-09-28，部署版本 9527b027，v1 模型）

输入：
> 我的体检指标：怀孕次数6，血浆葡萄糖148，血压72，皮褶厚度35，胰岛素0，BMI 33.6，糖尿病谱系函数0.627，年龄50。请评估我的糖尿病风险。

智能体输出（要点实录）：
> **糖尿病风险预测结果：52.64%，中风险**。智能体调用 GBDT 模型纯 Python 推理（参数文件 `models/gbdt_trees.json`，100 棵树，无需 sklearn 环境），结果与本地 sklearn GBDT 预测 0.5264 一致（误差 < 1e-8）；随后依次输出八项指标健康解读、就医建议、生活方式建议，并固定附带免责声明（本结果由机器学习模型基于 UCI Pima Indians Diabetes 数据集训练生成，仅供健康参考，不构成临床诊断建议）。

风险等级判定规则：<1/3 低风险，1/3~2/3 中风险，≥2/3 高风险。

v2 模型同样本本地复测：**73.33%**（胰岛素 0 按"未测"以训练集中位数填充后推理，
较 v1 将 0 当真实值更合理），判定为高风险/阳性（τ=0.375），
纯 Python 推理与 sklearn Pipeline 最大误差 3.3e-16。

![扣子智能体对话实测截图](docs/agent_demo_chat.jpg)
（截图为扣子编程项目页预览面板实拍：智能体回复结尾与固定免责声明部分。）

## ❓ 常见问题

1. **端口 3306 被占用**：在 `.env` 中修改 `DIABETES_DB_PORT`（如 3307）后重启。
2. **MySQL root 有密码 / 想改应用账号密码**：复制 `.env.example` 为 `.env` 修改后重跑 01→03 即可。
3. **PowerShell 提示禁止运行脚本**：用 `powershell -ExecutionPolicy Bypass -File 脚本路径.ps1`。
4. **重装依赖**：`.\venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`。
5. **matplotlib 中文乱码**：Windows 自带 SimHei/微软雅黑已配置（见 `03_train_models.py`）；Linux 需自行安装中文字体。
6. **体检某项没测**：血糖/舒张压/皮褶厚度/胰岛素/BMI 支持填 0 表示"未测"，模型按训练集中位数自动填充。

## 📚 数据来源与声明

- 数据来源：[UCI ML Repository — Pima Indians Diabetes Database](https://archive.ics.uci.edu/dataset/34/pima+indians+diabetes)（21 岁以上女性 Pima 族裔，768 例）；镜像文件与字段说明见 `data/raw/`。
- 本项目仅用于机器学习工程实践与教学演示，**平台输出不构成临床诊断依据**。
- 代码以 [MIT License](LICENSE) 开源。
