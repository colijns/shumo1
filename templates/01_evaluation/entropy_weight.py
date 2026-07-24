"""
模块：entropy_weight
功能：熵权法客观赋权——依据各指标取值的信息熵（离散程度）自动确定权重
适用题型：国赛C题高频 / 华数杯高频（客观赋权，常与 TOPSIS、模糊评价组合使用）
依赖：numpy
用法：直接运行 `python entropy_weight.py` 查看 demo；或 import 后调用 entropy_weight(X, types)
"""

import numpy as np


def entropy_weight(X, types=None):
    """
    熵权法求客观权重：指标取值差异越大（熵越小），提供的信息越多，权重越大。

    计算步骤：
        1. 正向化：成本型指标用 max - x 转化，效益型不变，再整体平移使各列非负；
        2. 归一化：p_ij = z_ij / Σ_i z_ij，得到第 j 个指标下第 i 个方案的比重；
        3. 熵值：e_j = -(1/ln n) Σ_i p_ij ln p_ij；
        4. 差异系数：d_j = 1 - e_j，归一化即得权重 w_j = d_j / Σ_k d_k。

    参数：
        X     : numpy.ndarray 或 pandas.DataFrame，(n 方案 × m 指标) 原始评价矩阵，n >= 2
        types : list[str] 或 None，指标类型（'benefit' 效益型 / 'cost' 成本型），
                None 表示全部按效益型处理
    返回：
        (m,) numpy.ndarray，各指标权重（和为 1）；函数内部会打印各指标熵值与权重
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X 必须是 (n 方案 × m 指标) 的二维矩阵')
    n, m = X.shape
    if n < 2:
        raise ValueError('熵权法至少需要 2 个方案（n >= 2）')
    if types is None:
        types = ['benefit'] * m
    if len(types) != m:
        raise ValueError(f'types 长度（{len(types)}）必须与指标数（{m}）一致')

    # 1. 正向化 + 非负平移
    Z = np.zeros_like(X)
    for j in range(m):
        if types[j] == 'cost':
            Z[:, j] = X[:, j].max() - X[:, j]      # 成本型：max - x
        elif types[j] == 'benefit':
            Z[:, j] = X[:, j] - X[:, j].min()     # 效益型：平移到非负
        else:
            raise ValueError(f"未知指标类型 '{types[j]}'，本文件支持 'benefit'/'cost'")

    # 2. 归一化求比重（加 eps 防止除零与 log(0)）
    eps = 1e-12
    col_sum = Z.sum(axis=0, keepdims=True)
    P = Z / (col_sum + eps)
    P = np.where(P <= 0, eps, P)

    # 3. 各指标熵值 e_j = -(1/ln n) Σ p ln p
    k = 1.0 / np.log(n)
    e = -k * np.sum(P * np.log(P), axis=0)

    # 4. 差异系数归一化得权重
    d = 1.0 - e
    if d.sum() <= 0:
        # 所有指标取值完全相同（熵均为 1）时退化为等权
        w = np.ones(m) / m
    else:
        w = d / d.sum()

    # 打印各指标熵值与权重
    print('----- 熵权法：各指标熵值与权重 -----')
    for j in range(m):
        print(f'指标{j + 1}：熵值 e = {e[j]:.4f}，差异系数 d = {d[j]:.4f}，权重 w = {w[j]:.4f}')
    return w


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：5 个方案 × 4 个指标（第 3 个为成本型）=====
    indicator_names = ['科研产出', '师资力量', '经费投入(成本)', '学生满意度']
    X = np.array([
        [85, 90, 5200, 88],
        [92, 85, 4750, 90],
        [78, 88, 6100, 82],
        [88, 92, 4400, 85],
        [95, 80, 5650, 93],
    ], dtype=float)
    types = ['benefit', 'benefit', 'cost', 'benefit']

    print('===== 熵权法客观赋权 demo（5 方案 × 4 指标）=====')
    print('原始评价矩阵：')
    print(X)
    print(f'指标类型：{types}\n')

    w = entropy_weight(X, types=types)

    print('\n----- 权重汇总 -----')
    for name, wj in zip(indicator_names, w):
        print(f'{name}：{wj:.4f}')
    print(f'权重之和 = {w.sum():.4f}')
    print('\nentropy_weight demo 运行完毕')
