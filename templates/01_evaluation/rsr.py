"""
模块：rsr
功能：秩和比法（RSR, Rank Sum Ratio）综合评价——编秩、计算秩和比、名次与概率单位 Probit
适用题型：国赛C题高频 / 华数杯高频（医疗卫生、工作质量等多指标综合评价）
依赖：numpy, scipy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python rsr.py` 查看 demo；或 import 后调用 rsr(X, w, types)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def rsr(X, w=None, types=None):
    """
    秩和比法综合评价。

    计算步骤：
        1. 编秩：对每个指标（列）把 n 个被评单位编秩，效益型按值升序（值大秩大），
           成本型按值降序（值小秩大），并列取平均秩；
        2. 秩和比：RSR_i = Σ_j w_j R_ij / n（w 和为 1；等权时退化为 Σ_j R_ij / (m·n)）；
        3. 名次：RSR 越大名次越靠前；
        4. Probit：将 RSR 升序排列，用 (i - 0.5) / n 估计累计频率 p，
           概率单位 Probit = Φ⁻¹(p) + 5（Φ 为标准正态分布函数）。

    参数：
        X     : numpy.ndarray 或 pandas.DataFrame，(n 单位 × m 指标) 原始评价矩阵
        w     : array_like 或 None，(m,) 指标权重，None 表示等权
        types : list[str] 或 None，指标类型（'benefit' 效益型 / 'cost' 成本型），
                None 表示全部为效益型
    返回：
        dict：
            rsr     : (n,) numpy.ndarray，各单位秩和比（越大越优）
            ranking : (n,) numpy.ndarray，各单位名次（1 为最优）
            probit  : (n,) numpy.ndarray，各单位 RSR 对应的概率单位
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X 必须是 (n 单位 × m 指标) 的二维矩阵')
    n, m = X.shape
    if n < 2:
        raise ValueError('秩和比法至少需要 2 个评价单位（n >= 2）')
    if types is None:
        types = ['benefit'] * m
    if len(types) != m:
        raise ValueError(f'types 长度（{len(types)}）必须与指标数（{m}）一致')

    # 1. 编秩：效益型升序编秩；成本型取负后升序编秩（等价于值越小秩越大）
    R = np.zeros((n, m))
    for j in range(m):
        col = X[:, j]
        if types[j] == 'cost':
            col = -col
        elif types[j] != 'benefit':
            raise ValueError(f"未知指标类型 '{types[j]}'，本文件支持 'benefit'/'cost'")
        R[:, j] = stats.rankdata(col, method='average')  # 并列取平均秩

    # 2. 权重处理
    if w is None:
        w_used = np.ones(m) / m
    else:
        w_used = np.asarray(w, dtype=float).ravel()
        if w_used.shape[0] != m:
            raise ValueError(f'权重长度（{w_used.shape[0]}）必须与指标数（{m}）一致')
        w_used = w_used / w_used.sum()

    # 3. 秩和比 RSR
    rsr_val = (R * w_used).sum(axis=1) / n

    # 4. 名次：RSR 越大名次越靠前
    ranking = rsr_val.argsort()[::-1].argsort() + 1

    # 5. Probit：RSR 升序 -> 累计频率 (i-0.5)/n -> Φ⁻¹(p) + 5
    order = rsr_val.argsort()                      # 升序排列的下标
    p = (np.arange(1, n + 1) - 0.5) / n            # 累计频率估计
    probit_sorted = stats.norm.ppf(p) + 5.0        # 概率单位
    probit = np.empty(n)
    probit[order] = probit_sorted                  # 还原到原始单位顺序

    return {'rsr': rsr_val, 'ranking': ranking, 'probit': probit}


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：医院工作质量综合评价（6 个单位 × 4 个指标）=====
    unit_names = [f'医院{chr(65 + i)}' for i in range(6)]  # 医院A ~ 医院F
    indicator_names = ['病床使用率%(效益)', '平均住院日(成本)', '治愈率%(效益)', '病死率%(成本)']
    X = np.array([
        [94.2, 11.5, 86.3, 1.8],
        [91.4, 12.1, 84.5, 2.0],
        [96.0, 10.8, 88.9, 1.5],
        [89.7, 13.0, 82.1, 2.3],
        [93.5, 11.2, 87.0, 1.7],
        [90.8, 12.5, 83.6, 2.1],
    ])
    types = ['benefit', 'cost', 'benefit', 'cost']

    print('===== 秩和比法（RSR）综合评价 demo：医院工作质量 =====')
    print(f'评价指标：{indicator_names}')
    print('原始数据：')
    print(X)

    res = rsr(X, w=None, types=types)

    print('\n----- 各医院 RSR 评价结果 -----')
    print(f'{"单位":<6}\t{"RSR":>8}\t{"Probit":>8}\t{"名次":>4}')
    for i, name in enumerate(unit_names):
        print(f'{name:<6}\t{res["rsr"][i]:>8.4f}\t{res["probit"][i]:>8.4f}\t{res["ranking"][i]:>4}')

    best = int(np.argmax(res['rsr']))
    print(f'\n结论：工作质量最优的是 {unit_names[best]}（RSR = {res["rsr"][best]:.4f}）')

    # ---- 可视化：RSR 柱状图（按 RSR 降序展示）----
    order = np.argsort(res['rsr'])[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([unit_names[i] for i in order], res['rsr'][order],
           color='#4C72B0', edgecolor='black')
    for k, idx in enumerate(order):
        ax.text(k, res['rsr'][idx] + 0.005, f'{res["rsr"][idx]:.4f}',
                ha='center', fontsize=10)
    ax.set_title('各医院工作质量秩和比（RSR）排序')
    ax.set_ylabel('RSR 值')
    ax.set_ylim(0, max(res['rsr']) * 1.2)
    plt.tight_layout()
    fig.savefig('rsr_demo.png', dpi=150, bbox_inches='tight')
    print('rsr_demo.png 已保存')
    plt.close(fig)
    print('\nrsr demo 运行完毕')
