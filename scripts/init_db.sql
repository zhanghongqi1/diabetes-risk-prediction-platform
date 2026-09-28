-- ============================================================
-- 糖尿病风险预测平台 - 数据库初始化脚本
-- MySQL 8.4 LTS / utf8mb4
-- 可重复执行（IF NOT EXISTS / CREATE USER IF NOT EXISTS）
-- ============================================================

CREATE DATABASE IF NOT EXISTS diabetes_platform
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

CREATE USER IF NOT EXISTS 'diabetes_app'@'localhost'
  IDENTIFIED BY 'Diabetes@2026';
GRANT ALL PRIVILEGES ON diabetes_platform.* TO 'diabetes_app'@'localhost';
FLUSH PRIVILEGES;

USE diabetes_platform;

-- 1) 原始体检数据表（UCI Pima Indians Diabetes，768 条）
CREATE TABLE IF NOT EXISTS patient_raw (
  id                          INT PRIMARY KEY COMMENT '记录序号(1-768)',
  pregnancies                 INT            NOT NULL COMMENT '怀孕次数',
  glucose                     INT                      COMMENT '口服糖耐量2小时血糖(mg/dL)',
  blood_pressure              INT                      COMMENT '舒张压(mm Hg)',
  skin_thickness              INT                      COMMENT '三头肌皮褶厚度(mm)',
  insulin                     INT                      COMMENT '2小时血清胰岛素(mu U/mL)',
  bmi                         DECIMAL(6,2)             COMMENT '体质指数 kg/m^2',
  diabetes_pedigree_function  DECIMAL(7,4)             COMMENT '糖尿病谱系功能值',
  age                         INT                      COMMENT '年龄(岁)',
  outcome                     TINYINT        NOT NULL COMMENT '是否糖尿病: 1=是 0=否'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='UCI Pima 糖尿病原始数据';

-- 2) 清洗后训练数据表（异常0值按中位数填充）
CREATE TABLE IF NOT EXISTS patient_clean (
  id                          INT PRIMARY KEY,
  pregnancies                 FLOAT NOT NULL,
  glucose                     FLOAT NOT NULL,
  blood_pressure              FLOAT NOT NULL,
  skin_thickness              FLOAT NOT NULL,
  insulin                     FLOAT NOT NULL,
  bmi                         FLOAT NOT NULL,
  diabetes_pedigree_function  FLOAT NOT NULL,
  age                         FLOAT NOT NULL,
  outcome                     TINYINT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='清洗后训练数据';

-- 3) 风险预测记录表（单例查询 / 批量导入留痕）
CREATE TABLE IF NOT EXISTS prediction_records (
  id                          INT AUTO_INCREMENT PRIMARY KEY,
  batch_no                    VARCHAR(40)     COMMENT '批量导入批次号',
  pregnancies                 FLOAT,
  glucose                     FLOAT,
  blood_pressure              FLOAT,
  skin_thickness              FLOAT,
  insulin                     FLOAT,
  bmi                         FLOAT,
  diabetes_pedigree_function  FLOAT,
  age                         FLOAT,
  actual_outcome              TINYINT         COMMENT '实际标签(模型评估时填写)',
  risk_probability            DECIMAL(6,4)    COMMENT '预测为糖尿病的概率',
  risk_level                  VARCHAR(10)     COMMENT '风险等级: 低/中/高',
  predicted_class             TINYINT         COMMENT '预测类别 1/0',
  model_name                  VARCHAR(30)     COMMENT '使用的模型',
  created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='风险预测记录';

-- 4) 模型信息与评估指标表
CREATE TABLE IF NOT EXISTS model_info (
  model_id        INT AUTO_INCREMENT PRIMARY KEY,
  model_name      VARCHAR(30) NOT NULL COMMENT 'LR/SVM/GBDT',
  accuracy        DECIMAL(6,4),
  precision_score DECIMAL(6,4),
  recall_score    DECIMAL(6,4),
  f1_score        DECIMAL(6,4),
  auc             DECIMAL(6,4),
  cv_auc_mean     DECIMAL(6,4) COMMENT '五折交叉验证AUC均值',
  cv_auc_std      DECIMAL(6,4) COMMENT '五折交叉验证AUC标准差',
  model_path      VARCHAR(255) COMMENT 'joblib模型文件相对路径',
  is_selected     TINYINT DEFAULT 0 COMMENT '1=当前选定上线模型',
  trained_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='模型训练评估记录';
