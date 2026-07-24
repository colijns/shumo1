# -*- coding: utf-8 -*-
"""
问题1(1) 无储能运行仿真核心逻辑
参考实现：Q1/milp_model.py -> solve_park_optimization(P_ess=0, E_ess=0)
"""
import pulp

def solve_no_storage(load, G_pv, G_w, C_PV=0.4, C_W=0.5, C_G=1.0):
    """风光优先直供 → 购电 → 弃电。MILP 统一口径。"""
    T = 24
    prob = pulp.LpProblem("NoStorage", pulp.LpMinimize)

    P_load_pv = [pulp.LpVariable(f'ld_pv_{t}', lowBound=0) for t in range(T)]
    P_load_w  = [pulp.LpVariable(f'ld_w_{t}', lowBound=0) for t in range(T)]
    P_curt_pv = [pulp.LpVariable(f'curt_pv_{t}', lowBound=0) for t in range(T)]
    P_curt_w  = [pulp.LpVariable(f'curt_w_{t}', lowBound=0) for t in range(T)]
    P_grid    = [pulp.LpVariable(f'grid_{t}', lowBound=0) for t in range(T)]

    for t in range(T):
        prob += P_load_pv[t] + P_curt_pv[t] == G_pv[t]       # 光伏分配
        prob += P_load_w[t]  + P_curt_w[t]  == G_w[t]        # 风电分配
        prob += P_load_pv[t] + P_load_w[t] + P_grid[t] == load[t]  # 负荷平衡

    prob += pulp.lpSum([
        C_PV * P_load_pv[t] + C_W * P_load_w[t] + C_G * P_grid[t]
        for t in range(T)
    ])
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return prob.status == pulp.LpStatusOptimal
