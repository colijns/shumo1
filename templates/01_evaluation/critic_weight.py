"""
模块：critic_weight
功能：CRITIC 客观赋权——综合指标的对比强度（标准差）与冲突性（相关系数）确定权重
适用题型：国赛C题高频 / 华数杯高频（客观赋权，可与熵权法对比/组合）
依赖：numpy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python critic_weight.py` 查看 demo；或 import 后调用 critic_weight(X)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def critic_weight(X):
    """
    CRITIC（Criteria Importance Through Intercriteria Correlation）赋权法。

    基本思想：指标权重由两部分信息量决定——
        对比强度 σ_j：指标取值的标准差，波动越大分辨力越强；
        冲突性   R_j = Σ_k (1 - r_jk)：与其他指标相关性越低，信息越独立。
        综合信息量 C_j = σ_j × R_j，权重 w_j = C_j / Σ_k C_k。

    参数：
        X : numpy.ndarray 或 pandas.DataFrame，(n 方案 × m 指标) 评价矩阵，
            建议输入前已完成指标正向化（如含成本型指标，先统一转为越大越好）
    返回：
        (m,) numpy.ndarray，各指标权重（和为 1）
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X 必须是 (n 方案 × m 指标) 的二维矩阵')
    n, m = X.shape
    if n < 2:
        raise ValueError('CRITIC 赋权至少需要 2 个方案（n >= 2）')

    # 1. 极差归一化（无量纲化，同时保留各指标波动差异）
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    span = np.where(x_max - x_min == 0, 1.0, x_max - x_min)  # 防止除零
    Z = (X - x_min) / span

    # 2. 对比强度：各指标标准差
    sigma = Z.std(axis=0, ddof=0)                # (m,)

    # 3. 冲突性：R_j = Σ_k (1 - r_jk)，r 为 Pearson 相关系数
    if m == 1:
        R = np.ones(1)                           # 单指标无冲突可言
    else:
        corr = np.corrcoef(Z, rowvar=False)      # (m, m) 相关系数矩阵
        corr = np.nan_to_num(corr, nan=0.0)      # 常数列会产生 NaN，按 0 相关处理
        R = np.sum(1.0 - corr, axis=1)           # (m,)

    # 4. 信息量 C_j = σ_j × R_j，归一化得权重
    C = sigma * R
    if C.sum() <= 0:
        w = np.ones(m) / m                       # 退化情形：等权
    else:
        w = C / C.sum()

    # 打印中间量，便于论文中展示计算过程
    print('----- CRITIC 赋权：中间量与权重 -----')
    for j in range(m):
        print(f'指标{j + 1}：标准差 σ = {sigma[j]:.4f}，冲突性 R = {R[j]:.4f}，'
              f'信息量 C = {C[j]:.4f}，权重 w = {w[j]:.4f}')
    return w


def _entropy_weight(X):
    """内部熵权法（仅 demo 对比用，保证单文件独立，假定输入已全部为效益型）。"""
    n, m = X.shape
    Z = X - X.min(axis=0, keepdims=True)
    eps = 1e-12
    P = Z / (Z.sum(axis=0, keepdims=True) + eps)
    P = np.where(P <= 0, eps, P)
    e = -(1.0 / np.log(n)) * np.sum(P * np.log(P), axis=0)
    d = 1.0 - e
    return d / d.sum() if d.sum() > 0 else np.ones(m) / m


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：6 个方案 × 4 个指标（全部为效益型），与熵权法对比 =====
    indicator_names = ['科研水平', '教学质量', '就业情况', '社会声誉']
    X = np.array([
        [85, 90, 78, 88],
        [92, 85, 82, 90],
        [78, 88, 90, 82],
        [88, 92, 75, 85],
        [95, 80, 85, 93],
        [80, 86, 88, 80],
    ], dtype=float)

    print('===== CRITIC 赋权 demo（6 方案 × 4 指标）=====')
    print('原始评价矩阵（全部为效益型指标）：')
    print(X)

    w_critic = critic_weight(X)
    w_entropy = _entropy_weight(X)

    # ---- 两种客观赋权结果对比表 ----
    print('\n----- CRITIC 权重 vs 熵权法权重 -----')
    print(f'{"指标":<8}\t{"CRITIC":>8}\t{"熵权法":>8}')
    for name, wc, we in zip(indicator_names, w_critic, w_entropy):
        print(f'{name:<8}\t{wc:>8.4f}\t{we:>8.4f}')

    # ---- 可视化：权重对比分组柱状图 ----
    x = np.arange(len(indicator_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, w_critic, width, label='CRITIC 权重', color='#4C72B0', edgecolor='black')
    ax.bar(x + width / 2, w_entropy, width, label='熵权法权重', color='#DD8452', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(indicator_names)
    ax.set_ylabel('权重')
    ax.set_title('CRITIC 赋权与熵权法结果对比')
    ax.legend()
    plt.tight_layout()
    fig.savefig('critic_weight_demo.png', dpi=150, bbox_inches='tight')
    print('critic_weight_demo.png 已保存')
    plt.close(fig)
    print('\ncritic_weight demo 运行完毕')
