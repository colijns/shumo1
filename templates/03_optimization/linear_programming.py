# -*- coding: utf-8 -*-
"""
模块：linear_programming
功能：线性规划（LP）求解，基于 scipy.optimize.linprog（HiGHS 求解器）
适用题型：通用最高频（生产计划、资源分配、运输调配、配料问题等）
依赖：numpy, scipy
用法：直接运行 `python linear_programming.py` 查看 demo；或 import 后调用 solve_lp
"""

import numpy as np
from scipy.optimize import linprog


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    求解标准线性规划：min c^T x，满足 A_ub x <= b_ub，A_eq x = b_eq，bounds 限定变量上下界。

    参数：
        c      : 目标函数系数向量，shape (n,)
        A_ub   : 不等式约束系数矩阵，shape (m_ub, n)，可为 None
        b_ub   : 不等式约束右端向量，shape (m_ub,)，可为 None
        A_eq   : 等式约束系数矩阵，shape (m_eq, n)，可为 None
        b_eq   : 等式约束右端向量，shape (m_eq,)，可为 None
        bounds : 每个变量的 (下界, 上界) 序列，None 表示默认 (0, +inf)

    返回：
        dict，键为：
            'x'       : 最优解向量（失败时为 None）
            'fun'     : 最优目标值（失败时为 None）
            'success' : 是否求解成功（bool）
            'message' : 求解器返回的状态信息
    """
    c = np.asarray(c, dtype=float)
    # method='highs' 为 scipy 新版默认求解器，速度快且稳定
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    result = {
        'x': res.x if res.success else None,
        'fun': res.fun if res.success else None,
        'success': bool(res.success),
        'message': str(res.message),
    }
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("线性规划 demo：经典生产计划问题")
    print("=" * 60)
    # 问题描述：
    #   某厂生产甲、乙两种产品，每件利润分别为 3 元、5 元；
    #   资源约束：设备工时 1*x1 + 2*x2 <= 8；原材料 A：4*x1 <= 16；原材料 B：3*x2 <= 12
    #   求利润最大的生产方案。
    # linprog 只能求最小化，因此利润最大化等价于 min -(3*x1 + 5*x2)
    c = [-3, -5]                       # 最大化利润 -> 取负转为最小化
    A_ub = [[1, 2],                    # 设备工时约束
            [4, 0],                    # 原材料 A 约束
            [0, 3]]                    # 原材料 B 约束
    b_ub = [8, 16, 12]
    bounds = [(0, None), (0, None)]    # 产量非负

    res = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if res['success']:
        x = res['x']
        max_profit = -res['fun']       # 还原为最大利润
        print("求解状态：成功")
        print(f"最优生产方案：甲产品 {x[0]:.4f} 件，乙产品 {x[1]:.4f} 件")
        print(f"最大利润：{max_profit:.4f} 元")
        # 校验约束使用情况（资源消耗量）
        used = np.dot(np.array(A_ub, dtype=float), x)
        print(f"资源消耗校验：设备工时 {used[0]:.4f}/8，"
              f"原材料A {used[1]:.4f}/16，原材料B {used[2]:.4f}/12")
    else:
        print(f"求解失败：{res['message']}")

    # 附加示例：带等式约束的营养配餐问题（最小化成本）
    print("-" * 60)
    print("附加示例：营养配餐（成本最小，含等式约束）")
    # 两种食品单位成本 4、3；要求总重量恰好 5 kg；蛋白质 >= 20 单位
    c2 = [4, 3]
    A_ub2 = [[-6, -5]]                 # 蛋白质 >= 20  ->  -6*x1 - 5*x2 <= -20
    b_ub2 = [-20]
    A_eq2 = [[1, 1]]                   # 总重量 = 5
    b_eq2 = [5]
    res2 = solve_lp(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2)
    if res2['success']:
        print(f"最优配餐：食品1 {res2['x'][0]:.4f} kg，食品2 {res2['x'][1]:.4f} kg")
        print(f"最小成本：{res2['fun']:.4f} 元")
    else:
        print(f"求解失败：{res2['message']}")
