# -*- coding: utf-8 -*-
"""
模块：multi_objective
功能：多目标优化的两种简化求解方法——加权求和法与理想点法（对方案集进行综合评价与排序）
适用题型：通用高频（多目标决策、方案优选、帕累托解集筛选后的最终决策）
依赖：numpy
用法：直接运行 `python multi_objective.py` 查看 demo；或 import 后调用 weighted_sum / ideal_point
"""

import numpy as np


def weighted_sum(F, weights):
    """
    加权求和法：将多目标合成为单一评价值，对候选方案排序（内部先做极差归一化以消除量纲）。

    参数：
        F       : 目标值矩阵，shape (n_samples, n_objectives)，每列为一个目标
        weights : 各目标权重，shape (n_objectives,)，和为 1

    返回：
        dict，键为：
            'scores'  : 各方案综合得分（越大越好），shape (n_samples,)
            'ranking' : 排名（第 1 名最优），shape (n_samples,)，整数数组
    """
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()          # 权重归一化，防止用户输入和不为 1

    # 极差归一化到 [0,1]（越大越好方向），防止不同量纲支配加权和
    f_min, f_max = F.min(axis=0), F.max(axis=0)
    span = np.where(f_max - f_min > 1e-12, f_max - f_min, 1.0)
    F_norm = (F - f_min) / span

    scores = F_norm @ weights
    # 排名：得分最高的为第 1 名（argsort 降序后反查名次）
    ranking = np.empty(len(scores), dtype=int)
    ranking[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return {'scores': scores, 'ranking': ranking}


def ideal_point(F, types):
    """
    理想点法（TOPSIS 思想的简化版）：以各目标最优值构成正理想点，按到理想点的距离排序。

    参数：
        F     : 目标值矩阵，shape (n_samples, n_objectives)
        types : 各目标类型列表，取值 'max'（越大越好）或 'min'（越小越好）

    返回：
        dict，键为：
            'scores'      : 各方案贴近度得分（越大越好），shape (n_samples,)
            'ranking'     : 排名（第 1 名最优），shape (n_samples,)，整数数组
            'ideal'       : 正理想点向量
            'anti_ideal'  : 负理想点向量
    """
    F = np.asarray(F, dtype=float)
    assert len(types) == F.shape[1], "types 长度必须等于目标个数"

    # 1. 极差归一化，并把成本型目标翻转为效益型，统一为「越大越好」
    F_norm = F.copy()
    for j in range(F.shape[1]):
        f_min, f_max = F[:, j].min(), F[:, j].max()
        span = f_max - f_min if f_max - f_min > 1e-12 else 1.0
        col = (F[:, j] - f_min) / span
        if types[j] == 'min':
            col = 1 - col                     # 成本型翻转为效益型
        F_norm[:, j] = col

    # 2. 正/负理想点
    ideal = F_norm.max(axis=0)
    anti_ideal = F_norm.min(axis=0)

    # 3. 各方案到正、负理想点的欧氏距离
    d_pos = np.sqrt(((F_norm - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((F_norm - anti_ideal) ** 2).sum(axis=1))

    # 4. 贴近度：距负理想点越远、距正理想点越近越好
    scores = d_neg / (d_pos + d_neg + 1e-12)
    ranking = np.empty(len(scores), dtype=int)
    ranking[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return {'scores': scores, 'ranking': ranking,
            'ideal': ideal, 'anti_ideal': anti_ideal}


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现

    print("=" * 60)
    print("多目标优化 demo：双目标（成本 + 质量）方案排序")
    print("=" * 60)
    # 6 个候选方案，两个目标：
    #   目标1：成本（万元，越小越好 min）
    #   目标2：质量评分（越大越好 max）
    plans = ['方案A', '方案B', '方案C', '方案D', '方案E', '方案F']
    F = np.array([
        [80, 90],     # 方案A：成本高、质量好
        [60, 75],     # 方案B：较均衡
        [50, 60],     # 方案C：便宜但质量一般
        [70, 85],     # 方案D
        [90, 95],     # 方案E：最贵、质量最好
        [55, 70],     # 方案F
    ], dtype=float)

    # ---------- 方法一：加权求和法 ----------
    # 先将成本型目标取负转为「越大越好」，再做归一化加权
    F_benefit = F.copy()
    F_benefit[:, 0] = -F_benefit[:, 0]        # 成本取负 -> 越大越好
    weights = [0.4, 0.6]                       # 决策者更看重质量
    res1 = weighted_sum(F_benefit, weights)
    print("【加权求和法】权重：成本 0.4，质量 0.6")
    for i, name in enumerate(plans):
        print(f"  {name}：成本 {F[i, 0]:.0f} 万，质量 {F[i, 1]:.0f} 分，"
              f"综合得分 {res1['scores'][i]:.4f}，排名第 {res1['ranking'][i]}")
    best1 = np.argmin(res1['ranking'])
    print(f"  => 加权求和法推荐：{plans[best1]}")

    # ---------- 方法二：理想点法 ----------
    res2 = ideal_point(F, types=['min', 'max'])
    print("-" * 60)
    print("【理想点法】目标类型：成本 min，质量 max")
    print(f"  正理想点：{np.round(res2['ideal'], 4)}，负理想点：{np.round(res2['anti_ideal'], 4)}")
    for i, name in enumerate(plans):
        print(f"  {name}：贴近度 {res2['scores'][i]:.4f}，排名第 {res2['ranking'][i]}")
    best2 = np.argmin(res2['ranking'])
    print(f"  => 理想点法推荐：{plans[best2]}")

    print("-" * 60)
    agree = "一致" if best1 == best2 else "不一致（可结合权重灵敏度进一步分析）"
    print(f"两种方法结论是否一致：{agree}")
