"""
模块：topsis
功能：TOPSIS（逼近理想解排序法）综合评价，以及与熵权法组合的熵权-TOPSIS
适用题型：国赛C题最高频 / 华数杯最高频（多指标方案排序与综合评价）
依赖：numpy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python topsis.py` 查看 demo；或 import 后调用 topsis() / entropy_topsis()
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def _forward_transform(X, types):
    """
    内部指标正向化（保证单文件独立，不跨目录 import）：
    成本型用 max - x 转化为效益型，效益型保持不变。
    """
    X = np.asarray(X, dtype=float).copy()
    if types is None:
        return X
    for j, t in enumerate(types):
        if t == 'cost':
            X[:, j] = X[:, j].max() - X[:, j]
        elif t == 'benefit':
            pass
        else:
            raise ValueError(f"未知指标类型 '{t}'，本文件支持 'benefit'/'cost'")
    return X


def _entropy_weight(Z):
    """
    内部熵权法定权（保证单文件独立）：
    输入已正向化的矩阵 Z，返回 (m,) 权重向量。
    """
    n, m = Z.shape
    Z2 = Z - Z.min(axis=0, keepdims=True)          # 平移为非负
    eps = 1e-12
    P = Z2 / (Z2.sum(axis=0, keepdims=True) + eps)
    P = np.where(P <= 0, eps, P)
    e = -(1.0 / np.log(n)) * np.sum(P * np.log(P), axis=0)  # 各指标熵值
    d = 1.0 - e                                              # 差异系数
    if d.sum() <= 0:
        return np.ones(m) / m
    return d / d.sum()


def topsis(X, w=None, types=None):
    """
    TOPSIS 综合评价：以各方案到正/负理想解的相对贴近度作为得分。

    计算步骤：
        1. 指标正向化（成本型 -> 效益型）；
        2. 向量归一化：z_ij = x_ij / sqrt(Σ_i x_ij²)；
        3. 加权：v_ij = w_j * z_ij；
        4. 正理想解 V+ = 各列最大值，负理想解 V- = 各列最小值；
        5. 贴近度得分 C_i = D_i- / (D_i+ + D_i-)，越大越优。

    参数：
        X     : numpy.ndarray 或 pandas.DataFrame，(n 方案 × m 指标) 原始评价矩阵
        w     : array_like 或 None，(m,) 指标权重，None 表示等权
        types : list[str] 或 None，指标类型（'benefit'/'cost'），None 全为效益型
    返回：
        dict：
            scores       : (n,) numpy.ndarray，各方案贴近度得分（越大越优）
            ranking      : (n,) numpy.ndarray，各方案名次（1 为最优）
            weights_used : (m,) numpy.ndarray，实际使用的归一化权重
    """
    Z = _forward_transform(X, types)
    n, m = Z.shape

    # 1. 向量归一化
    norm = np.sqrt((Z ** 2).sum(axis=0, keepdims=True))
    norm = np.where(norm == 0, 1.0, norm)        # 防止除零
    Z = Z / norm

    # 2. 权重处理：等权或归一化用户权重
    if w is None:
        w_used = np.ones(m) / m
    else:
        w_used = np.asarray(w, dtype=float).ravel()
        if w_used.shape[0] != m:
            raise ValueError(f'权重长度（{w_used.shape[0]}）必须与指标数（{m}）一致')
        w_used = w_used / w_used.sum()

    # 3. 加权规范化矩阵
    V = Z * w_used

    # 4. 正、负理想解
    v_pos = V.max(axis=0)
    v_neg = V.min(axis=0)

    # 5. 欧氏距离与贴近度
    d_pos = np.sqrt(((V - v_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - v_neg) ** 2).sum(axis=1))
    scores = d_neg / (d_pos + d_neg + 1e-12)

    # 6. 排名：得分越高名次越靠前（第 1 名最优）
    ranking = scores.argsort()[::-1].argsort() + 1

    return {'scores': scores, 'ranking': ranking, 'weights_used': w_used}


def entropy_topsis(X, types=None):
    """
    熵权-TOPSIS 组合评价：先用熵权法客观定权，再做 TOPSIS 排序。

    参数：
        X     : numpy.ndarray 或 pandas.DataFrame，(n 方案 × m 指标) 原始评价矩阵
        types : list[str] 或 None，指标类型（'benefit'/'cost'），None 全为效益型
    返回：
        dict：结构与 topsis() 相同，{'scores', 'ranking', 'weights_used'}
    """
    Z = _forward_transform(X, types)
    w = _entropy_weight(Z)
    return topsis(X, w=w, types=types)


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：6 个供应商方案 × 4 个指标（含 2 个成本型）=====
    indicator_names = ['产品质量', '产品价格(成本)', '交付周期(成本)', '售后服务']
    scheme_names = [f'方案{i + 1}' for i in range(6)]
    X = np.array([
        [85, 3000, 12, 90],
        [90, 3200, 10, 85],
        [78, 2800, 15, 92],
        [88, 3500, 11, 80],
        [92, 3100,  9, 88],
        [80, 2600, 14, 95],
    ], dtype=float)
    types = ['benefit', 'cost', 'cost', 'benefit']

    # ===== IO 用法示例：合成数据写 CSV -> load_table 读回 -> 调用 =====
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common'))
    from io_utils import export_result, load_table
    demo_df_path = 'topsis_demo_data.csv'
    export_result(pd.DataFrame(X, columns=indicator_names), demo_df_path)
    X_loaded = load_table(demo_df_path).values
    print('IO 演示：从 CSV 读回的数据形状 =', X_loaded.shape)
    os.remove(demo_df_path)

    print('===== TOPSIS 综合评价 demo（6 方案 × 4 指标）=====')
    print('原始评价矩阵：')
    print(X)
    print(f'指标类型：{types}')

    # ---- 1. 等权 TOPSIS ----
    res_eq = topsis(X, w=None, types=types)
    print('\n----- 等权 TOPSIS 结果 -----')
    for i, name in enumerate(scheme_names):
        print(f'{name}：得分 = {res_eq["scores"][i]:.4f}，名次 = {res_eq["ranking"][i]}')

    # ---- 2. 熵权-TOPSIS ----
    res_et = entropy_topsis(X, types=types)
    print('\n----- 熵权-TOPSIS 结果 -----')
    print('熵权法自动定权：')
    for name, wj in zip(indicator_names, res_et['weights_used']):
        print(f'  {name}：w = {wj:.4f}')
    print('综合得分与排名：')
    for i, name in enumerate(scheme_names):
        print(f'{name}：得分 = {res_et["scores"][i]:.4f}，名次 = {res_et["ranking"][i]}')

    best = int(np.argmax(res_et['scores']))
    print(f'\n结论：熵权-TOPSIS 下最优方案为 {scheme_names[best]}'
          f'（得分 {res_et["scores"][best]:.4f}）')

    # ----- 可视化：熵权-TOPSIS 得分柱状图 -----
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#DD8452' if r == 1 else '#4C72B0' for r in res_et['ranking']]
    ax.bar(scheme_names, res_et['scores'], color=colors, edgecolor='black')
    for i, s in enumerate(res_et['scores']):
        ax.text(i, s + 0.005, f'{s:.4f}', ha='center', fontsize=9)
    ax.set_title('熵权-TOPSIS 综合得分（橙色为最优方案）')
    ax.set_xlabel('方案')
    ax.set_ylabel('贴近度得分')
    ax.set_ylim(0, max(res_et['scores']) * 1.2)
    plt.tight_layout()
    fig.savefig('topsis_demo.png', dpi=150, bbox_inches='tight')
    print('已保存图：topsis_demo.png')
    print('\ntopsis demo 运行完毕')
