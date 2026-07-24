# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

数学建模比赛仓库。包含完整代码模板库（`templates/`）和当前赛题工作区（按题号划分子文件夹，代码从 templates 复制出来编写）。

## 环境与基本命令

- **Python 环境**: `math` (conda) — 每个新 shell 先 `conda activate math`
- **安装依赖**: `pip install -r requirements.txt` (核心: numpy/scipy/pandas/matplotlib/seaborn/scikit-learn/statsmodels/networkx/pytest)
- **可选依赖**: PuLP（整数规划）、pyecharts（交互式地图）— 用到再装
- **运行 demo**: `python <模块路径>`（每个 `.py` 自带合成数据 demo）
- **跑测试**: `python -m pytest tests/ -q`（全量）、`python -m pytest tests/test_<模块>.py -v`（单模块）
- **按名筛选测试**: `python -m pytest tests/ -k <关键词>`
- **GBK 编码问题**: 先 `conda activate math` 再 `python xxx.py`，不要用 `conda run -n math python xxx.py`

## 目录结构

```
D:\code_warehouse\Projects\shumo1/
├── templates/               # 数学建模代码模板库（不可直接修改，需复制出来用）
│   ├── 01_evaluation/       # 综合评价（TOPSIS/AHP/熵权/模糊/PCA/RSR）
│   ├── 02_prediction/       # 预测（灰色/指数平滑/回归/马尔可夫/BP/ARIMA）
│   ├── 03_optimization/     # 优化（LP/IP/NLP/多目标/GA/PSO/SA）
│   ├── 04_statistics/       # 统计（描述性/相关/插值/蒙特卡洛/假设检验）
│   ├── 05_clustering_classification/  # 聚类分类（KMeans/层次/决策树/SVM）
│   ├── 06_graph_network/    # 图论网络（最短路/MST/最大流）
│   ├── 07_simulation_ode/   # 仿真与ODE（Logistic/SIR/LV/差分/排队论）
│   ├── 08_plot_templates/   # 论文绘图模板（折线/柱状/热力/雷达/3D/地图）
│   ├── common/              # 通用工具（data_utils/metrics/io_utils/plot_style）
│   ├── USAGE.md             # 完整使用指南
│   └── MODULES.md           # 模块速查表（AI 适配用，每模块一行）
├── docs/
│   └── 赛题结构化拆解.md     # 当前赛题的问题分析
├── attachment/              # 赛题附件数据
├── <题号>/                  # 各题代码目录（如 Q1/, Q2/ 等）
└── requirements.txt
```

## 模板模块速查大类

| 题型 | 模块目录 | 典型场景 |
|------|----------|----------|
| 评价 | `01_evaluation/` | 多指标方案排序、主观/客观赋权、模糊评价 |
| 预测 | `02_prediction/` | 小样本灰色预测、时间序列ARIMA、回归、BP |
| 优化 | `03_optimization/` | 线性/整数/非线性规划、GA/PSO/SA、多目标 |
| 统计 | `04_statistics/` | EDA、相关性、插值拟合、蒙特卡洛、假设检验 |
| 聚类分类 | `05_clustering_classification/` | KMeans、层次、决策树、SVM |
| 图论 | `06_graph_network/` | 最短路径、MST、最大流 |
| 仿真 | `07_simulation_ode/` | ODE求解、差分方程、排队论 |
| 绘图 | `08_plot_templates/` | 论文配图（折线/柱状/热力/雷达/3D/地图） |
| 通用 | `common/` | 数据预处理、精度评价、IO、绘图样式 |

## 开发约定

1. **单文件独立**: 每个模块文件自包含，不 import 其他算法模块。复制单文件即可在比赛代码中使用
2. **函数返回 dict**: 带语义化键（如 `{'scores', 'ranking', 'weights_used'}`），不返回裸 tuple
3. **随机函数带 seed 参数**: `seed=None` 真随机，`seed=int` 可复现。比赛传固定 seed（如 `seed=42`）
4. **绘图用 savefig 不用 plt.show**: 已设 `matplotlib.use('Agg')` + 中文字体，headless 友好
5. **数据 IO 用 `load_table` / `export_result`**: 在 `common/io_utils.py`，按扩展名自动识别 csv/xlsx
6. **正向化/归一化/缺失值填充用 `common/data_utils.py`**
7. **精度评价用 `common/metrics.py`**: RMSE/MAE/MAPE/R²
8. **路径引入**: `sys.path.append('01_evaluation')` 后 `from topsis import entropy_topsis`

## 当前赛题

当前赛题为微电网风光储配置优化题（题目在 `docs/赛题结构化拆解.md`）。附件数据在 `attachment/` 下：
- `附件1：各园区典型日负荷数据.xlsx`
- `附件2：各园区典型日风光发电数据.xlsx`
- `附件3：12个月各园区典型日风光发电数据.xlsx`

## 工作流

1. 判题型 → 扫 `templates/MODULES.md` 选模块
2. 跑 demo 确认模块功能 → `conda activate math && python <模块路径>`
3. 复制模板到题号目录 → 把 demo 数据换成赛题数据（经 `load_table` 读取）
4. 出论文配图 → `08_plot_templates/`
5. 固定所有随机 seed
6. 跑测试确认接口未破坏
