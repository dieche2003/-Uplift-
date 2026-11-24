# 基于 Uplift 建模的广告投放增量效果分析

机器学习课程项目，使用 Criteo Uplift v2.1 随机广告实验数据，比较传统访问预测与广告增量排序，并通过 ROI 敏感性模拟分析投放决策。

## 项目流程

1. `preprocess.py`：校验原始数据、输出 EDA、按 treatment 分层抽样30%，并按7:1:2划分训练集、验证集和测试集。
2. `models.py --stage baselines`：训练逻辑回归和 LightGBM 访问预测基线，仅使用 `f0~f11`。
3. `models.py --stage uplift`：使用 scikit-uplift 的 SoloModel（S-learner）和 TwoModels（T-learner）训练 uplift 模型。
4. `evaluate.py`：计算测试集 AUC、Qini 系数和归一化 AUUC，并绘制 Qini 曲线。
5. `evaluate.py --stage roi`：比较 Top-K uplift 定向投放与随机投放，生成 ROI 敏感性结果。

传统模型预测 `visit`，输入不含 `treatment` 或 `exposure`。Uplift 分析以 `treatment` 作为随机分配变量，采用 ITT 口径；`exposure` 表示实际曝光，只用于 ROI 成本估计。AUC、Qini/AUUC 和 ROI 分别衡量预测排序、增量排序和假设成本下的业务收益，不应相互替代。

## 环境与运行

在 Windows PowerShell 中进入项目目录。依赖可使用以下命令安装：

```powershell
cd D:\xiangmu-guanggao
python -m pip install pandas numpy scikit-learn lightgbm joblib pyarrow scikit-uplift matplotlib
```

确认原始数据位于 `D:\xiangmu-guanggao\raw\criteo-uplift-v2.1.csv`，然后按顺序运行：

```powershell
python -X utf8 -u preprocess.py
python -X utf8 -u models.py --stage baselines
python -X utf8 -u models.py --stage uplift
python -X utf8 -u evaluate.py
python -X utf8 -u evaluate.py --stage roi
```

各阶段可单独调用；运行后会更新相应的 `data`、`model`、`result` 和 `figure` 文件。

## 目录说明

| 路径 | 内容 |
|---|---|
| `raw/` | 原始 Criteo CSV 数据 |
| `data/` | 抽样数据及 train、validation、test Parquet 划分 |
| `model/` | LR、LightGBM 基线及 S/T-learner 模型 |
| `result/` | 预测分数、指标表、ROI 情景结果与元数据 |
| `figure/` | Qini 曲线和 ROI 敏感性图 |
| `preprocess.py` | 数据预处理与 EDA |
| `models.py` | 传统预测基线及 uplift 模型训练 |
| `evaluate.py` | 模型评估、作图及 ROI 模拟 |

## 主要测试结果

| 模型 | 测试指标 |
|---|---:|
| LR | AUC = 0.931666 |
| LightGBM 预测基线 | AUC = 0.947133 |
| S-learner | Qini = 0.075560；归一化 AUUC = 0.029861 |
| T-learner | Qini = 0.069827；归一化 AUUC = 0.027577 |

ROI 分析覆盖 Top-5%、10%、20%、30%，访问价值 `{5, 10, 15}` 与单次曝光成本 `{0.5, 1, 2}`，共108组情景。参考情景（访问价值10、曝光成本1）下，Top-5% S-learner、T-learner 和随机策略的模拟 ROI 分别为154.65%、180.46%和157.68%。ROI 是测试集上的点估计模拟值，不代表线上真实收益；成本按每个曝光阳性样本计费一次，且定向与随机策略按相同目标人数比较，并非相同预算。

## 主要输出

- `result/model_comparison_metrics.csv`：基线 AUC 与 uplift 指标汇总。
- `result/uplift_test_predictions.parquet`：测试集 uplift 预测分数。
- `result/roi_sensitivity_results.csv`：ROI 敏感性分析情景表。
- `result/roi_business_analysis.md`：业务分析结果与口径说明。
- `figure/qini_curve.png`、`figure/roi_sensitivity.png`：论文用图。

