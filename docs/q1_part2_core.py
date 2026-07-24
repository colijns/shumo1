# -*- coding: utf-8 -*-
"""
问题1(2)(3) 核心代码片段 —— 含储能运行优化与容量配置 MILP
参考实现：Q1/milp_model.py -> solve_park_optimization(...)

核心逻辑：
  - (2) 固定储能：给定 P_ess=50, E_ess=100，求最优运行策略
  - (3) 优化容量：P_ess/E_ess 为决策变量，同时求最优容量与运行策略
  - 工程整数优化：capacity_steps 约束容量为步长的整数倍（如 5kW/10kWh）
"""

import numpy as np
import pulp

# 参数（与 Q1/data_loader.py 一致）
C_PV = 0.4       # 光伏购电成本
C_W = 0.5        # 风电购电成本
C_G = 1.0        # 电网购电价格
C_P_ESS = 800    # 储能功率单价 元/kW
C_E_ESS = 1800   # 储能容量单价 元/kWh
Y = 10           # 储能寿命 年
ETA_C = 0.95     # 充电效率
ETA_D = 0.95     # 放电效率
S_MIN, S_MAX = 0.10, 0.90  # SOC 上下限
S_0 = 0.50       # 初始 SOC
DT = 1.0         # 时段长度 h


def solve_storage_optimization(
    load,           # array[24] 负荷 kW
    G_pv,           # array[24] 光伏出力 kW
    G_w,            # array[24] 风电出力 kW
    P_ess=0,        # 储能额定功率 kW
    E_ess=0,        # 储能额定容量 kWh
    optimize_capacity=False,   # True 时 P_ess/E_ess 为决策变量
    capacity_steps=None,       # (P_step, E_step) 工程整数约束
    capacity_bounds=None,      # (P_max, E_max) 容量上界
    verbose=False,
):
    """
    统一 MILP 模型：无储能 / 固定储能 / 优化容量
    """
    T = 24
    big_M = max(np.max(load), np.max(G_pv) + np.max(G_w), 1000)

    prob = pulp.LpProblem("Storage_Optimization", pulp.LpMinimize)

    # ----- 容量变量 -----
    if optimize_capacity:
        P_up, E_up = capacity_bounds or (big_M, big_M)
        if capacity_steps is None:
            # 连续容量
            P_ess_var = pulp.LpVariable('P_ess', lowBound=0, upBound=P_up)
            E_ess_var = pulp.LpVariable('E_ess', lowBound=0, upBound=E_up)
            actual_P = P_ess_var
            actual_E = E_ess_var
        else:
            # 工程整数容量（如 5kW/10kWh 步长）
            P_step, E_step = capacity_steps
            nP = pulp.LpVariable('nP', lowBound=0,
                                 upBound=int(P_up // P_step),
                                 cat=pulp.LpInteger)
            nE = pulp.LpVariable('nE', lowBound=0,
                                 upBound=int(E_up // E_step),
                                 cat=pulp.LpInteger)
            actual_P = P_step * nP
            actual_E = E_step * nE
    else:
        actual_P = P_ess
        actual_E = E_ess

    # ----- 时段变量 -----
    P_load_pv = [pulp.LpVariable(f'ld_pv_{t}', lowBound=0) for t in range(T)]
    P_load_w  = [pulp.LpVariable(f'ld_w_{t}', lowBound=0) for t in range(T)]
    P_ch_pv   = [pulp.LpVariable(f'ch_pv_{t}', lowBound=0) for t in range(T)]
    P_ch_w    = [pulp.LpVariable(f'ch_w_{t}', lowBound=0) for t in range(T)]
    P_curt_pv = [pulp.LpVariable(f'curt_pv_{t}', lowBound=0) for t in range(T)]
    P_curt_w  = [pulp.LpVariable(f'curt_w_{t}', lowBound=0) for t in range(T)]
    P_ch      = [pulp.LpVariable(f'ch_{t}', lowBound=0) for t in range(T)]
    P_dis     = [pulp.LpVariable(f'dis_{t}', lowBound=0) for t in range(T)]
    E         = [pulp.LpVariable(f'E_{t}', lowBound=0) for t in range(T + 1)]
    z         = [pulp.LpVariable(f'z_{t}', cat=pulp.LpBinary) for t in range(T)]
    P_grid    = [pulp.LpVariable(f'grid_{t}', lowBound=0) for t in range(T)]

    # ----- 约束 -----
    for t in range(T):
        # 光伏分配：直供负荷 + 充电 + 弃光 = 出力
        prob += P_load_pv[t] + P_ch_pv[t] + P_curt_pv[t] == G_pv[t]
        # 风电分配：直供负荷 + 充电 + 弃风 = 出力
        prob += P_load_w[t] + P_ch_w[t] + P_curt_w[t] == G_w[t]
        # 总充电 = 光伏充电 + 风电充电
        prob += P_ch[t] == P_ch_pv[t] + P_ch_w[t]
        # 负荷平衡：风光直供 + 放电 + 网购电 = 负荷
        prob += P_load_pv[t] + P_load_w[t] + P_dis[t] + P_grid[t] == load[t]

    for t in range(T):
        # SOC 递推（含充放电效率）
        prob += E[t+1] == E[t] + ETA_C * P_ch[t] * DT - P_dis[t] * DT / ETA_D
        # SOC 上下限
        prob += E[t] >= S_MIN * actual_E
        prob += E[t] <= S_MAX * actual_E
        # 充放电功率上限
        prob += P_ch[t] <= actual_P
        prob += P_dis[t] <= actual_P
        # 互斥约束（禁止同时充放电）
        prob += P_ch[t] <= big_M * z[t]
        prob += P_dis[t] <= big_M * (1 - z[t])
    # 日末 SOC 回归
    prob += E[T] >= S_MIN * actual_E
    prob += E[T] <= S_MAX * actual_E
    prob += E[T] == E[0]
    # 初始 SOC
    prob += E[0] == S_0 * actual_E

    # ----- 目标函数 -----
    daily_oper = pulp.lpSum([
        C_PV * (P_load_pv[t] + P_ch_pv[t]) * DT +
        C_W  * (P_load_w[t] + P_ch_w[t]) * DT +
        C_G  * P_grid[t] * DT
        for t in range(T)
    ])
    if optimize_capacity:
        annual_inv = (C_P_ESS * actual_P + C_E_ESS * actual_E) / Y
        prob += 365 * daily_oper + annual_inv, "annual_total_cost"
    else:
        prob += daily_oper, "daily_operating_cost"

    # ----- 求解 -----
    prob.solve(pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=120))
    if prob.status != pulp.LpStatusOptimal:
        return {'success': False, 'status': pulp.LpStatus[prob.status]}

    def val(v): return pulp.value(v) or 0.0

    # ----- 提取结果 -----
    result = {
        'success': True, 'P_ess': val(actual_P), 'E_ess': val(actual_E),
        'daily_cost': val(daily_oper),
        'P_ch': [val(P_ch[t]) for t in range(T)],
        'P_dis': [val(P_dis[t]) for t in range(T)],
        'E': [val(E[t]) for t in range(T + 1)],
        'P_grid': [val(P_grid[t]) for t in range(T)],
        'P_curt_pv': [val(P_curt_pv[t]) for t in range(T)],
        'P_curt_w': [val(P_curt_w[t]) for t in range(T)],
    }

    # 派生指标
    result['load_total'] = float(np.sum(load) * DT)
    result['grid_total'] = float(np.sum(result['P_grid']) * DT)
    result['pv_curt'] = float(np.sum(result['P_curt_pv']) * DT)
    result['w_curt'] = float(np.sum(result['P_curt_w']) * DT)
    pv_use = float((np.sum(G_pv) - np.sum(result['P_curt_pv'])) * DT)
    w_use = float((np.sum(G_w) - np.sum(result['P_curt_w'])) * DT)
    re_gen = float(np.sum(G_pv) * DT) + float(np.sum(G_w) * DT)
    result['re_ratio'] = (pv_use + w_use) / re_gen if re_gen > 0 else 0.0
    result['inv_annual'] = (C_P_ESS * result['P_ess'] +
                            C_E_ESS * result['E_ess']) / Y
    result['annual_cost'] = 365 * result['daily_cost'] + result['inv_annual']
    result['unit_annual_cost'] = result['annual_cost'] / (
        365 * result['load_total']) if result['load_total'] > 0 else 0

    return result


# ===== 快捷入口 =====
def run_no_storage(load, pv, w):
    return solve_storage_optimization(load, pv, w, P_ess=0, E_ess=0)

def run_fixed_storage(load, pv, w, P_ess=50, E_ess=100):
    return solve_storage_optimization(load, pv, w, P_ess=P_ess, E_ess=E_ess)

def run_optimized_storage(load, pv, w):
    return solve_storage_optimization(load, pv, w, optimize_capacity=True)

def run_engineering_storage(load, pv, w, P_step=5, E_step=10, P_max=200, E_max=600):
    return solve_storage_optimization(
        load, pv, w, optimize_capacity=True,
        capacity_steps=(P_step, E_step),
        capacity_bounds=(P_max, E_max),
    )


# ===== 快速验证 =====
def _demo():
    load = np.array([200, 180, 170, 160, 180, 250, 350, 420,
                     400, 380, 360, 350, 340, 360, 380, 370,
                     360, 400, 420, 380, 320, 280, 240, 200])
    pv = np.array([0, 0, 0, 10, 80, 200, 400, 550,
                   600, 580, 500, 400, 350, 300, 250, 200,
                   150, 80, 20, 0, 0, 0, 0, 0])
    w = np.zeros(24)

    print("无储能:")
    r0 = run_no_storage(load, pv, w)
    if r0['success']:
        print(f"  日成本 {r0['daily_cost']:.2f} 购电 {r0['grid_total']:.1f}")

    print("固定 50kW/100kWh:")
    r1 = run_fixed_storage(load, pv, w)
    if r1['success']:
        print(f"  日成本 {r1['daily_cost']:.2f} 年综合 {r1['annual_cost']:.2f}")

    print("工程整数最优:")
    r2 = run_engineering_storage(load, pv, w)
    if r2['success']:
        print(f"  {r2['P_ess']:.0f}kW/{r2['E_ess']:.0f}kWh  年综合 {r2['annual_cost']:.2f}")


if __name__ == '__main__':
    _demo()
