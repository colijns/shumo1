# -*- coding: utf-8 -*-
"""
问题1(1) 核心代码片段 —— 无储能运行仿真
参考实现：Q1/milp_model.py -> solve_park_optimization(P_ess=0, E_ess=0)

核心逻辑：风光优先直供负荷，不足从电网购电，多余弃电。
与问题1(2)(3)使用统一 MILP 模型，确保成本口径一致。
"""

import numpy as np
import pulp

# 参数（与 Q1/data_loader.py 一致）
C_PV = 0.4       # 光伏购电成本 元/kWh
C_W = 0.5        # 风电购电成本 元/kWh
C_G = 1.0        # 电网购电价格 元/kWh
ETA_C = 0.95     # 充电效率（仅用于统一接口，无储能时闲置）
ETA_D = 0.95     # 放电效率
DT = 1.0         # 时段长度 h


def solve_no_storage(load, G_pv, G_w):
    """
    无储能运行仿真（MILP 统一口径）

    参数：
        load : array[24]  负荷功率 kW
        G_pv : array[24]  光伏出力 kW
        G_w  : array[24]  风电出力 kW

    返回：
        dict: 各项经济指标
    """
    T = 24
    prob = pulp.LpProblem("NoStorage", pulp.LpMinimize)

    # ---------- 决策变量 ----------
    P_load_pv = [pulp.LpVariable(f'ld_pv_{t}', lowBound=0) for t in range(T)]
    P_load_w  = [pulp.LpVariable(f'ld_w_{t}', lowBound=0) for t in range(T)]
    P_curt_pv = [pulp.LpVariable(f'curt_pv_{t}', lowBound=0) for t in range(T)]
    P_curt_w  = [pulp.LpVariable(f'curt_w_{t}', lowBound=0) for t in range(T)]
    P_grid    = [pulp.LpVariable(f'grid_{t}', lowBound=0) for t in range(T)]

    # ---------- 约束 ----------
    for t in range(T):
        # 光伏出力分配：直供 + 弃光 = 出力
        prob += P_load_pv[t] + P_curt_pv[t] == G_pv[t]
        # 风电出力分配：直供 + 弃风 = 出力
        prob += P_load_w[t] + P_curt_w[t] == G_w[t]
        # 负荷平衡：风光直供 + 网购电 = 负荷
        prob += P_load_pv[t] + P_load_w[t] + P_grid[t] == load[t]

    # ---------- 目标：日运行成本最小 ----------
    daily_cost = pulp.lpSum([
        C_PV * P_load_pv[t] * DT +
        C_W * P_load_w[t] * DT +
        C_G * P_grid[t] * DT
        for t in range(T)
    ])
    prob += daily_cost

    # ---------- 求解 ----------
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status != pulp.LpStatusOptimal:
        return {'success': False, 'status': pulp.LpStatus[prob.status]}

    def val(v): return pulp.value(v) or 0.0

    # ---------- 派生指标 ----------
    result = {
        'success': True,
        'daily_cost': val(daily_cost),
        'load_total': float(np.sum(load) * DT),
        'grid_total': float(np.sum([val(P_grid[t]) for t in range(T)]) * DT),
        'pv_curt': float(np.sum([val(P_curt_pv[t]) for t in range(T)]) * DT),
        'w_curt': float(np.sum([val(P_curt_w[t]) for t in range(T)]) * DT),
        'pv_gen': float(np.sum(G_pv) * DT),
        'w_gen': float(np.sum(G_w) * DT),
    }
    pv_use = float(np.sum([val(P_load_pv[t]) for t in range(T)]) * DT)
    w_use = float(np.sum([val(P_load_w[t]) for t in range(T)]) * DT)
    re_use = pv_use + w_use
    re_gen = result['pv_gen'] + result['w_gen']
    result['re_ratio'] = re_use / re_gen if re_gen > 0 else 0.0
    result['unit_daily_cost'] = result['daily_cost'] / result['load_total']
    # 年综合成本 = 365 x 日运行成本（无储能时没有投资项）
    result['annual_cost'] = 365 * result['daily_cost']
    result['unit_annual_cost'] = result['annual_cost'] / (365 * result['load_total'])

    return result


# ===== 快速验证 =====
def _demo():
    load = np.array([200, 180, 170, 160, 180, 250, 350, 420,
                     400, 380, 360, 350, 340, 360, 380, 370,
                     360, 400, 420, 380, 320, 280, 240, 200])
    pv = np.array([0, 0, 0, 10, 80, 200, 400, 550,
                   600, 580, 500, 400, 350, 300, 250, 200,
                   150, 80, 20, 0, 0, 0, 0, 0])
    w = np.zeros(24)
    r = solve_no_storage(load, pv, w)
    if r['success']:
        print(f"日运行成本: {r['daily_cost']:.2f} 元")
        print(f"购电量: {r['grid_total']:.1f} kWh | 弃光: {r['pv_curt']:.1f} kWh")
        print(f"风光消纳率: {r['re_ratio']*100:.1f}%")


if __name__ == '__main__':
    _demo()
