# -*- coding: utf-8 -*-
"""
问题1(2)(3) 含储能 MILP 优化核心逻辑
参考实现：Q1/milp_model.py -> solve_park_optimization(...)
"""
import pulp

def solve_storage_optimization(load, G_pv, G_w,
                               P_ess=0, E_ess=0,
                               optimize_capacity=False,
                               capacity_steps=None,
                               capacity_bounds=None,
                               C_PV=0.4, C_W=0.5, C_G=1.0,
                               C_P_ESS=800, C_E_ESS=1800, Y=10,
                               ETA_C=0.95, ETA_D=0.95,
                               S_MIN=0.10, S_MAX=0.90, S_0=0.50):
    """统一 MILP：无储能(P=0,E=0) / 固定储能 / 优化容量(含工程整数)"""
    T, DT = 24, 1.0
    big_M = max(max(load), max(G_pv) + max(G_w), 1000)
    prob = pulp.LpProblem("StorageOpt", pulp.LpMinimize)

    # 容量变量
    if optimize_capacity:
        P_up, E_up = capacity_bounds or (big_M, big_M)
        if capacity_steps:
            P_step, E_step = capacity_steps
            nP = pulp.LpVariable('nP', lowBound=0, upBound=int(P_up//P_step), cat='Integer')
            nE = pulp.LpVariable('nE', lowBound=0, upBound=int(E_up//E_step), cat='Integer')
            actual_P, actual_E = P_step * nP, E_step * nE
        else:
            actual_P = pulp.LpVariable('P_ess', lowBound=0, upBound=P_up)
            actual_E = pulp.LpVariable('E_ess', lowBound=0, upBound=E_up)
    else:
        actual_P, actual_E = P_ess, E_ess

    # 时段变量
    P_ch = [pulp.LpVariable(f'ch_{t}', lowBound=0) for t in range(T)]
    P_dis = [pulp.LpVariable(f'dis_{t}', lowBound=0) for t in range(T)]
    E_ess_t = [pulp.LpVariable(f'E_{t}', lowBound=0) for t in range(T + 1)]
    z = [pulp.LpVariable(f'z_{t}', cat='Binary') for t in range(T)]
    P_grid = [pulp.LpVariable(f'grid_{t}', lowBound=0) for t in range(T)]
    P_load_pv = [pulp.LpVariable(f'ld_pv_{t}', lowBound=0) for t in range(T)]
    P_load_w = [pulp.LpVariable(f'ld_w_{t}', lowBound=0) for t in range(T)]

    for t in range(T):
        # 风光分配 + 负荷平衡
        prob += P_load_pv[t] + P_ch[t] <= G_pv[t]   # 光伏: 直供 + 充电 ≤ 出力
        prob += P_load_w[t] + P_ch[t] <= G_w[t]     # 风电: 直供 + 充电 ≤ 出力
        prob += P_load_pv[t] + P_load_w[t] + P_dis[t] + P_grid[t] == load[t]
        # SOC 递推
        prob += E_ess_t[t+1] == E_ess_t[t] + ETA_C * P_ch[t] * DT - P_dis[t] * DT / ETA_D
        prob += E_ess_t[t] >= S_MIN * actual_E
        prob += E_ess_t[t] <= S_MAX * actual_E
        prob += P_ch[t] <= actual_P
        prob += P_dis[t] <= actual_P
        prob += P_ch[t] <= big_M * z[t]            # 充放互斥
        prob += P_dis[t] <= big_M * (1 - z[t])
    prob += E_ess_t[T] == E_ess_t[0]                # 日末回归
    prob += E_ess_t[0] == S_0 * actual_E

    # 目标：年综合成本最小（含运行 + 投资分摊）
    daily_op = pulp.lpSum([C_PV*(P_load_pv[t]+P_ch[t]) + C_W*(P_load_w[t]+P_ch[t])
                           + C_G*P_grid[t] for t in range(T)]) * DT
    if optimize_capacity:
        prob += 365 * daily_op + (C_P_ESS * actual_P + C_E_ESS * actual_E) / Y
    else:
        prob += daily_op

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return prob.status == pulp.LpStatusOptimal
