# 使用指南

> 面向参赛者的实操教程。读完能:搭好环境、跑通 demo、在自己的比赛代码里复用任意模块、出论文配图、避开本机常见坑。
>
> 本库另有三份文档,分工如下:
> - **README.md** -- 速查表与概览(「题型 -> 模块」对照、模块明细)
> - **MODULES.md** -- AI 适配用的精简速查表(每模块一行:函数签名 / 输入输出形状 / 依赖)
> - **CLAUDE.md** -- 给 AI 助手的开发约定(改代码时读)
> - **本文 USAGE.md** -- 从零上手到比赛实战的完整工作流

---

## 1. 环境搭建(约 10 分钟)

本库所有脚本在 conda 环境 `math` 下运行。**切勿用 base 环境跑**,依赖版本已锁定在 `requirements.txt`。

### 1.1 创建 math 环境

```powershell
# 已装 Miniconda/Anaconda 后(PowerShell 或 Git Bash 均可)
conda create -n math python=3.11 -y
conda activate math
```

> Python 3.9+ 即可,推荐 3.11。`requirements.txt` 里 numpy 2.2 / scipy 1.15 等较新,3.11 兼容性最好。

### 1.2 安装依赖

```powershell
conda activate math            # 每个新 shell 先激活(一次)
pip install -r requirements.txt
```

核心依赖:numpy / scipy / pandas / matplotlib / seaborn / scikit-learn / statsmodels / networkx / pytest。版本已用 `==` 锁定,与作者环境一致。

### 1.3 可选依赖(用到再装)

```powershell
pip install PuLP               # 03_optimization/integer_programming.py(整数规划)
pip install pyecharts          # 08_plot_templates/maps_3d.py(交互式中国地图)
```

未装时对应模块的 demo 会打印一行提示并优雅跳过(退出码 0),不影响其他模块。

### 1.4 验证安装

```powershell
conda activate math
python 01_evaluation/topsis.py              # 跑某模块 demo,看输出 + 出图
python -m pytest tests/ -q                  # 全库 smoke 测试,应 167 passed / 2 skipped
```

`2 skipped` = PuLP / pyecharts 未装,属正常。看到 `167 passed` 即环境就绪。

> ⚠️ 跑 demo 用 `conda activate math` + `python xxx.py`,**不要用 `conda run -n math python xxx.py`**。原因见 [FAQ §9.2](#92-conda-run-跑脚本报-gbk-编码错误)。`conda run -n math python -m pytest tests/`(跑模块)则没问题。

---

## 2. 三分钟上手

```powershell
conda activate math
python 01_evaluation/topsis.py
```

你会看到:熵权-TOPSIS 在合成数据上的得分与排名打印,以及一张 `topsis_demo.png` 配图。每个 `.py` 文件都是这样的「自带合成数据 demo」,文件头四行 docstring 标注了【模块 / 功能 / 适用题型 / 依赖 / 用法】。

换任意模块同理:`python 03_optimization/simulated_annealing.py`、`python 07_simulation_ode/ode_solver.py` ...

---

## 3. 在比赛代码中复用模块

### 3.1 两种复用方式

**方式 A -- 路径引入(推荐,便于随库更新)**

```python
import sys, numpy as np
sys.path.append('01_evaluation')
from topsis import entropy_topsis

X = np.array([[85, 70, 0.8],
              [90, 65, 0.6],
              [78, 80, 0.9]], dtype=float)
types = ['benefit', 'benefit', 'cost']      # 第三列为成本型(越小越好)
res = entropy_topsis(X, types)
print(res['scores'])                         # 得分
print(res['ranking'])                        # 排名(1 = 最优)
```

**方式 B -- 拷贝单文件(最贴合「拷进 submission 独立运行」)**

把 `01_evaluation/topsis.py` 复制到你的比赛目录,直接 `from topsis import entropy_topsis`。**每个模块文件都是自包含的** -- 它不 import 其他算法模块,也不 import `common/`,小工具函数都在文件内以 `_` 前缀私有实现。所以单文件拷过去就能跑。

### 3.2 三条核心约定(复用前必知)

1. **函数返回 `dict`**(带语义化键,如 `{'scores', 'ranking', 'weights_used'}`),不返回裸 tuple。用 `res['xxx']` 取结果。
2. **随机函数带 `seed` 参数**:`seed=None` = 真随机,`seed=int` = 可复现。**比赛论文里务必传固定 seed**(如 `seed=42`),否则结果无法复核。
3. **绘图模块用 `savefig` 不用 `plt.show()`**:headless 友好,无显示器也能出图。demo 里已设 `matplotlib.use('Agg')` + 中文字体。

### 3.3 完整工作流:综合评价题

「读赛题 Excel -> 熵权-TOPSIS 排序 -> 导出结果 CSV -> 出图」一条龙:

```python
import sys, numpy as np
sys.path.append('common')
sys.path.append('01_evaluation')
from io_utils import load_table, export_result        # 数据 IO(供比赛代码调用)
from topsis import entropy_topsis

df = load_table('data.xlsx')                          # 自动读 csv/xlsx
num = df.select_dtypes('number').values               # 取数值列 -> ndarray
types = ['benefit', 'cost', 'benefit', 'benefit']     # 按列指定,顺序对齐
res = entropy_topsis(num, types)

# 导出结果(Excel 可直接打开)
export_result({
    '方案': df.iloc[:, 0].values,
    '得分': res['scores'],
    '排名': res['ranking'],
}, 'result_topsis.csv')
```

### 3.4 完整工作流:预测题

```python
import sys, numpy as np
sys.path.append('02_prediction')
from grey_model import gm11

x0 = np.array([71.1, 72.4, 74.5, 76.8, 79.3])         # 至少 4 个历史值
res = gm11(x0, n_predict=3)                            # 预测未来 3 期
print('预测值:', res['predict'])
print('精度等级:', res['grade'])                       # '优'/'良'/'中'/'差'
print('后验差比 C:', res['C'], ' 小误差概率 P:', res['P'])  # C<0.35 且 P>0.95 为优
```

数据多(≥20 期)用 ARIMA(`02_prediction/arima_model.py` 的 `select_order_aic` 自动定阶),数据少用 GM(1,1)。

### 3.5 完整工作流:优化题

```python
import sys
sys.path.append('03_optimization')
from linear_programming import solve_lp

# 例:max 2x + 3y  s.t. x+y≤10, 2x+y≤12, x,y≥0
res = solve_lp(c=[-2, -3],                             # min -2x-3y(取负转最小化)
               A_ub=[[1, 1], [2, 1]], b_ub=[10, 12],
               bounds=[(0, None), (0, None)])
print('最优解 x =', res['x'])
print('最优值   =', -res['fun'])                       # 再取负还原最大值
```

非线性用 `nonlinear_programming.py`(SLSQP);整数/0-1 用 `integer_programming.py`(PuLP);复杂组合优化用 `genetic_algorithm.py` / `particle_swarm.py` / `simulated_annealing.py`(手写,均带 `seed`)。

---

## 4. 数据 IO(`common/io_utils.py`)

这两个函数**给比赛代码调用**,不被各算法模块 import(保持单文件独立):

| 函数 | 签名 | 返回 | 说明 |
|---|---|---|---|
| `load_table` | `(path, sheet_name=0)` | `DataFrame` | 按扩展名自动选 `read_csv` / `read_excel` |
| `export_result` | `(obj, path)` | `str`(绝对路径) | `obj` 为 DataFrame 或 dict-of-lists,扩展名决定 csv/xlsx |

```python
from common.io_utils import load_table, export_result

df = load_table('附件1.xlsx', sheet_name='Sheet1')     # 读赛题数据
# ... 算法处理 ...
export_result(df_result, '结果.xlsx')                  # 导出,Excel 直接打开
```

> 比赛数据常有标题行/单位行/中文列名。`load_table` 读回 DataFrame 后,用 `df.select_dtypes('number')` 取数值列再喂给算法模块(算法模块接受 ndarray)。

---

## 5. 论文绘图(`08_plot_templates/`)

三类模板,函数均返回 `matplotlib.figure.Figure`,你拿到后 `savefig`:

```python
import sys, numpy as np
sys.path.append('08_plot_templates')
from basic_plots import plot_line
from statistical_plots import plot_heatmap

x = np.arange(2015, 2025)
fig = plot_line(x, {'实际值': y_true, '预测值': y_pred})
fig.savefig('trend.png', dpi=200, bbox_inches='tight')   # 高清存图
```

| 文件 | 覆盖图型 |
|---|---|
| `basic_plots.py` | 折线 / 分组柱状 / 堆叠柱状 / 散点+回归 / 饼图 |
| `statistical_plots.py` | 热力图 / 箱线图 / 小提琴图 / 雷达图 |
| `maps_3d.py` | 3D 曲面 / 等高线 / pyecharts 交互式中国地图 |

中文字体已在每个绘图文件顶部内联设置(`SimHei` / `Microsoft YaHei` / `Noto Sans CJK SC` 回退),Windows 直接出中文,macOS/Linux 装 `Noto Sans CJK SC` 即可。

> 若只想统一绘图样式,可 `from common.plot_style import set_chinese_style, save_fig`。但绘图模块本身不依赖它(单文件独立)。

---

## 6. 可复现性

所有含随机的模块都遵循同一约定:

```python
res = kmeans_auto(X, k_range=range(2, 8), seed=None)   # 真随机,每次不同
res = kmeans_auto(X, k_range=range(2, 8), seed=42)     # 固定 seed,可复现
```

**比赛论文建议**:所有随机调用传同一 `seed=42`(或任意固定 int),并在论文里注明。评委复核时能拿到完全一致的结果。各模块 demo 默认 `seed=42`。

---

## 7. 测试(`tests/`)

每个模块配一个 `tests/test_<module>.py` smoke 测试,断言「能加载 / 输入输出结构 / 类型正确 / 数值有限」:

```powershell
conda activate math
python -m pytest tests/ -q                  # 全跑
python -m pytest tests/test_topsis.py -v    # 单测某模块
python -m pytest tests/ -k monte_carlo       # 按名筛选
```

改了某模块后,跑对应 `test_<module>.py` 快速验证没破坏接口。`tests/conftest.py` 提供路径式加载 fixture(支持 `01_` 前缀目录)。

---

## 8. 拿到题目后的选模块流程

1. **判题型** -- 对照 README「题型 -> 模块速查」定位大类(评价 / 预测 / 优化 / 统计 / 聚类 / 图论 / ODE / 绘图)。
2. **扫 MODULES.md** -- 在该大类里按「问题信号」列找匹配行,看核心函数与输入输出形状。
3. **跑 demo** -- `python <对应模块>.py`,确认它解决的就是你要的问题。
4. **改数据** -- 把 demo 里的合成数据换成赛题数据(经 `load_table` 读入)。
5. **出图** -- 用 `08_plot_templates/` 把结果画成论文配图,`savefig` 存盘。
6. **固定 seed** -- 检查所有随机调用传了固定 seed。
7. **跑测试** -- `python -m pytest tests/test_<module>.py` 确认接口未被你改动破坏。

---

## 9. 常见问题(FAQ)

### 9.1 中文乱码 / 方框

- **Windows**:代码已内置 `SimHei` / `Microsoft YaHei` 回退,正常不会乱码。若仍乱码,检查是否误删了文件顶部的 `plt.rcParams['font.sans-serif']` 设置。
- **macOS/Linux**:`pip install` 或系统包管理器装 `Noto Sans CJK SC` 字体,清 matplotlib 字体缓存后即可。

### 9.2 conda run 跑脚本报 GBK 编码错误

**现象**:`conda run -n math python 02_prediction/grey_model.py` 报
`UnicodeEncodeError: 'gbk' codec can't encode character ...`

**原因**:Windows 控制台默认 GBK,`conda run` 转发子进程 stdout 时用 GBK 编码,而模块 demo 打印的中文 / 数学符号(如 `²`、`≈`)超出 GBK 字符集。

**解决**(任选其一):
```powershell
# 方案 1(推荐):先激活再跑,stdout 直连控制台
conda activate math
python 02_prediction/grey_model.py

# 方案 2:强制 Python 用 UTF-8
$env:PYTHONUTF8=1
python 02_prediction/grey_model.py

# 方案 3:Git Bash 里先 chcp 切到 UTF-8 代码页
chcp 65001
```

> `conda run -n math python -m pytest tests/`(跑 pytest 模块)不受此影响 -- pytest 输出以英文为主,GBK 可编码。所以测试命令照常用 `conda run`。

### 9.3 ModuleNotFoundError: No module named 'xxx'

- `from topsis import ...` 报错 -> 没加 `sys.path.append('01_evaluation')`,或当前目录不对。模块是路径式加载,必须把所在目录加进 `sys.path`。
- `import pulp` / `import pyecharts` 报错 -> 可选依赖没装,`pip install PuLP` / `pip install pyecharts`(见 §1.3)。

### 9.4 plt.show() 没反应 / 远程跑不出图

本库绘图模块统一用 `matplotlib.use('Agg')` + `savefig`,**没有 `plt.show()`**。这是为了在无显示器的服务器 / 远程环境也能出图。看图去打开 `savefig` 存的 `.png` 文件。

### 9.5 算法报除零 / 输入形状错误

- 赛题数据常有常数列或全零列,模块内已做 `np.where(span == 0, 1.0, span)` 防除零。若仍报错,检查输入是否含 `NaN` / `inf`,先用 `common/data_utils.fill_missing` 填充。
- 形状不符会抛 `ValueError` 并附中文说明,按提示调整(常见:把 DataFrame 传成了需要 ndarray 的函数,用 `.values` 转一下)。

### 9.6 Windows 路径反斜杠

代码里路径用正斜杠 `/` 或原始字符串 `r'...'` 最稳。`load_table('data/附件1.xlsx')` 在 Windows 同样可用,pandas 会正确处理。

---

## 10. 文档导航

| 文档 | 看它做什么 |
|---|---|
| **USAGE.md**(本文) | 从零上手、复用套路、数据 IO、绘图、避坑 |
| **README.md** | 速查:题型->模块对照、各模块功能明细、依赖 |
| **MODULES.md** | AI / 人快速定位:每模块一行函数签名 + 输入输出形状 |
| **CLAUDE.md** | 改代码时的开发约定(单文件独立、seed、Agg、测试) |
| **references/REFERENCES.md** | 外部优质开源仓库与教程清单(非本库代码) |

---

*本库为竞赛学习整理模板,全部代码可自由修改使用;赛题论文需自行撰写建模与分析,各赛事均查重,切勿套用成品论文。*
