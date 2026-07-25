# -*- coding: utf-8 -*-
"""
MILP 统一优化模型 —— 基于 PuLP
功能：求解三个场景的储能运行/容量优化
     - 无储能: P_ess=0, E_ess=0
     - 固定储能: P_ess=50, E_ess=100
     - 优化容量: P_ess, E_ess 为决策变量
"""

import sys
import os

import numpy as np

# 添加 templates 到路径以复用 data_loader 的常量
sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_loader import (
    C_PV, C_W, C_G, C_P_ESS, C_E_ESS, Y,
    ETA_C, ETA_D, S_MIN, S_MAX, S_0, DT
)

try:
    import pulp
    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False
    print('警告: PuLP 未安装，MILP 模型不可用。请运行: pip install pulp')


def solve_park_optimization(
    load,           # array[24], 负荷 kW
    G_pv,           # array[24], 光伏出力 kW
    G_w,            # array[24], 风电出力 kW
    P_ess=0,        # 储能额定功率 kW
    E_ess=0,        # 储能额定容量 kWh
    optimize_capacity=False,  # 是否将 P_ess/E_ess 作为决策变量
    capacity_steps=None,  # 工程离散步长 (功率kW, 容量kWh)，None表示连续容量
    capacity_bounds=None,  # 容量上界 (功率kW, 容量kWh)
    verbose=False,   # 是否打印求解器日志
    big_M=None,     # 大 M 值
    charge_cost=None,  # (光伏充电价, 风电充电价)；None 表示与消纳同价 (C_PV, C_W)
    c_p_ess=None,   # 储能功率单价覆盖；None 表示用 data_loader.C_P_ESS
    c_e_ess=None,   # 储能容量单价覆盖；None 表示用 data_loader.C_E_ESS
    curt_penalty=0.0,  # 弃电惩罚因子 λ (元/kWh)；0 时退化为 View A (弃电免费)
):
    """
    求解单个园区的储能运行优化 MILP 模型。

    参数：
        load, G_pv, G_w : 24 时段数据
        P_ess, E_ess    : 固定储能容量（optimize_capacity=False 时使用）
        optimize_capacity: True 时 P_ess/E_ess 为决策变量
        capacity_steps  : 如 (5, 10)，将容量限制为工程步长的整数倍
        capacity_bounds : 如 (200, 600)，限制容量变量搜索上界
        verbose         : 是否打印求解器日志
        big_M           : 大 M 值，None 时自动计算
        charge_cost     : (c_pv, c_w)，充电能量单价；None 时与消纳同价 (C_PV, C_W)。
                          用于 Q2(2) 充电成本敏感性：解耦"充电绿电价"与"负荷绿电价"。
        c_p_ess, c_e_ess: 储能功率/容量单价覆盖；None 时用 data_loader 常量。
                          用于 Q2(2) 投资价格敏感性。默认值下与原行为完全一致。
        curt_penalty    : 弃电惩罚因子 λ (元/kWh)，对弃风弃光计价。
                          默认 0 = View A（弃电不计成本），Q1/Q2(1)/Q2(2) 无惩罚版行为不变。
                          >0 时目标函数加 λ·(P_curt_pv+P_curt_w)·Δt，daily_cost 含惩罚项；
                          另导出 daily_curt_penalty 与 daily_cost_pure（不含惩罚）。

    返回：
        dict，包含求解状态、变量值、各项指标
    """
    if not _HAS_PULP:
        return {'status': 'PuLP 未安装', 'success': False}

    T = 24
    t_range = range(T)

    # ----- 大 M 自动计算 -----
    if big_M is None:
        max_load = max(load) if len(load) > 0 else 0
        total_gen = max(G_pv) + max(G_w)
        big_M = max(total_gen, max_load, 1000)

    # ----- 建立问题 -----
    prob = pulp.LpProblem("Park_Storage_Optimization", pulp.LpMinimize)

    # ----- 决策变量 -----
    # 容量变量
    if optimize_capacity:
        P_upper, E_upper = capacity_bounds or (big_M, big_M)
        if capacity_steps is None:
            actual_P_ess = pulp.LpVariable('P_ess', lowBound=0, upBound=P_upper)
            actual_E_ess = pulp.LpVariable('E_ess', lowBound=0, upBound=E_upper)
        else:
            P_step, E_step = capacity_steps
            if P_step <= 0 or E_step <= 0:
                raise ValueError('capacity_steps 必须为正数')
            n_P = pulp.LpVariable(
                'n_P_ess', lowBound=0,
                upBound=int(P_upper // P_step), cat=pulp.LpInteger
            )
            n_E = pulp.LpVariable(
                'n_E_ess', lowBound=0,
                upBound=int(E_upper // E_step), cat=pulp.LpInteger
            )
            actual_P_ess = P_step * n_P
            actual_E_ess = E_step * n_E
    else:
        actual_P_ess = P_ess
        actual_E_ess = E_ess

    # 时段变量
    P_load_pv = [pulp.LpVariable(f'P_load_pv_{t}', lowBound=0) for t in t_range]
    P_load_w  = [pulp.LpVariable(f'P_load_w_{t}', lowBound=0) for t in t_range]
    P_ch_pv   = [pulp.LpVariable(f'P_ch_pv_{t}', lowBound=0) for t in t_range]
    P_ch_w    = [pulp.LpVariable(f'P_ch_w_{t}', lowBound=0) for t in t_range]
    P_curt_pv = [pulp.LpVariable(f'P_curt_pv_{t}', lowBound=0) for t in t_range]
    P_curt_w  = [pulp.LpVariable(f'P_curt_w_{t}', lowBound=0) for t in t_range]
    P_ch      = [pulp.LpVariable(f'P_ch_{t}', lowBound=0) for t in t_range]
    P_dis     = [pulp.LpVariable(f'P_dis_{t}', lowBound=0) for t in t_range]
    E         = [pulp.LpVariable(f'E_{t}', lowBound=0) for t in list(t_range) + [24]]
    z         = [pulp.LpVariable(f'z_{t}', cat=pulp.LpBinary) for t in t_range]
    P_grid    = [pulp.LpVariable(f'P_grid_{t}', lowBound=0) for t in t_range]

    # ----- 约束 -----

    # (1) 光伏出力分配: G_pv = P_load_pv + P_ch_pv + P_curt_pv
    for t in t_range:
        prob += P_load_pv[t] + P_ch_pv[t] + P_curt_pv[t] == G_pv[t], f'pv_balance_{t}'

    # (2) 风电出力分配: G_w = P_load_w + P_ch_w + P_curt_w
    for t in t_range:
        prob += P_load_w[t] + P_ch_w[t] + P_curt_w[t] == G_w[t], f'w_balance_{t}'

    # (3) 总充电功率等于风光充电之和
    for t in t_range:
        prob += P_ch[t] == P_ch_pv[t] + P_ch_w[t], f'ch_sum_{t}'

    # (4) 负荷功率平衡
    for t in t_range:
        prob += (P_load_pv[t] + P_load_w[t] + P_dis[t] + P_grid[t]
                 == load[t]), f'load_balance_{t}'

    # (5) 储能电量递推: E[t+1] = E[t] + eta_c * P_ch[t] * dt - P_dis[t] * dt / eta_d
    for t in t_range:
        prob += (E[t+1] == E[t] + ETA_C * P_ch[t] * DT - P_dis[t] * DT / ETA_D), f'soc_trans_{t}'

    # (6) SOC 上下限
    for t in list(t_range) + [24]:
        prob += E[t] >= S_MIN * actual_E_ess, f'soc_low_{t}'
        prob += E[t] <= S_MAX * actual_E_ess, f'soc_high_{t}'

    # (7) 初始 SOC
    prob += E[0] == S_0 * actual_E_ess, 'init_soc'

    # (8) 日末 SOC = 初始 SOC
    prob += E[24] == E[0], 'end_soc'

    # (9) 储能充放电功率上限
    for t in t_range:
        prob += P_ch[t] <= actual_P_ess, f'ch_max_{t}'
        prob += P_dis[t] <= actual_P_ess, f'dis_max_{t}'

    # (10) 禁止同时充放电（MILP 互斥约束）
    for t in t_range:
        prob += P_ch[t] <= big_M * z[t], f'mutex_ch_{t}'
        prob += P_dis[t] <= big_M * (1 - z[t]), f'mutex_dis_{t}'

    # ----- 成本参数（默认与 data_loader 常量一致，保证 Q1/Q2(1) 行为不变）-----
    # 充电能量单价：None 时与消纳同价；否则解耦，用于 Q2(2) 充电成本敏感性
    if charge_cost is None:
        c_ch_pv, c_ch_w = C_PV, C_W
    else:
        c_ch_pv, c_ch_w = charge_cost
    # 储能投资单价：None 时用 data_loader 常量，用于 Q2(2) 投资价格敏感性
    cp_ess = C_P_ESS if c_p_ess is None else c_p_ess
    ce_ess = C_E_ESS if c_e_ess is None else c_e_ess

    # ----- 目标函数 -----
    # 典型日运行成本：负荷绿电按 C_PV/C_W，充电绿电按 c_ch_*（默认相同）；
    # 弃电按 curt_penalty 计价（默认 0 = View A 弃电免费，Q1/Q2(1) 行为不变）。
    daily_cost = pulp.lpSum([
        C_PV * P_load_pv[t] * DT + c_ch_pv * P_ch_pv[t] * DT +
        C_W * P_load_w[t] * DT + c_ch_w * P_ch_w[t] * DT +
        C_G * P_grid[t] * DT +
        curt_penalty * (P_curt_pv[t] + P_curt_w[t]) * DT
        for t in t_range
    ])

    if optimize_capacity:
        # 年综合成本 = 365 * 日运行成本 + 年均投资
        annual_inv = (cp_ess * actual_P_ess + ce_ess * actual_E_ess) / Y
        prob += 365 * daily_cost + annual_inv, 'annual_total_cost'
    else:
        prob += daily_cost, 'daily_operating_cost'

    # ----- 求解 -----
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=120)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    success = prob.status == pulp.LpStatusOptimal
    obj_val = pulp.value(prob.objective)

    if not success:
        return {
            'status': status,
            'success': False,
            'objective': obj_val,
            'prob': prob,
        }

    # ----- 提取结果 -----
    def val(v):
        if isinstance(v, (pulp.LpVariable, pulp.LpAffineExpression)):
            return pulp.value(v) or 0.0
        return float(v)

    # 容量
    final_P_ess = val(actual_P_ess)
    final_E_ess = val(actual_E_ess)

    # 各时段变量
    result = {
        'status': status,
        'success': True,
        'P_ess': final_P_ess,
        'E_ess': final_E_ess,
        'objective': obj_val,
        't': list(range(T)),
        'P_load_pv': [val(P_load_pv[t]) for t in t_range],
        'P_load_w': [val(P_load_w[t]) for t in t_range],
        'P_ch_pv': [val(P_ch_pv[t]) for t in t_range],
        'P_ch_w': [val(P_ch_w[t]) for t in t_range],
        'P_curt_pv': [val(P_curt_pv[t]) for t in t_range],
        'P_curt_w': [val(P_curt_w[t]) for t in t_range],
        'P_ch': [val(P_ch[t]) for t in t_range],
        'P_dis': [val(P_dis[t]) for t in t_range],
        'E': [val(E[t]) for t in range(T + 1)],
        'z': [val(z[t]) for t in t_range],
        'P_grid': [val(P_grid[t]) for t in t_range],
    }

    # ----- 计算派生指标 -----
    result['daily_cost'] = val(daily_cost)
    result['load_total'] = float(np.sum(load) * DT)
    result['grid_total'] = float(np.sum(result['P_grid']) * DT)
    result['pv_use'] = float(np.sum(result['P_load_pv']) + np.sum(result['P_ch_pv'])) * DT
    result['w_use'] = float(np.sum(result['P_load_w']) + np.sum(result['P_ch_w'])) * DT
    result['pv_curt'] = float(np.sum(result['P_curt_pv'])) * DT
    result['w_curt'] = float(np.sum(result['P_curt_w'])) * DT
    result['curt_total'] = result['pv_curt'] + result['w_curt']
    result['pv_gen'] = float(np.sum(G_pv) * DT)
    result['w_gen'] = float(np.sum(G_w) * DT)
    total_re_gen = result['pv_gen'] + result['w_gen']
    total_re_use = result['pv_use'] + result['w_use']
    result['re_ratio'] = total_re_use / total_re_gen if total_re_gen > 0 else 0.0

    # 弃电惩罚分解（curt_penalty=0 时 daily_curt_penalty=0, daily_cost_pure=daily_cost）
    result['curt_penalty'] = float(curt_penalty)
    result['daily_curt_penalty'] = curt_penalty * result['curt_total']
    result['daily_cost_pure'] = result['daily_cost'] - result['daily_curt_penalty']

    # 年均投资（固定储能也有投资成本，此处统一计算用于年综合成本）
    result['inv_cost'] = (cp_ess * final_P_ess + ce_ess * final_E_ess)
    result['inv_annual'] = result['inv_cost'] / Y

    # 年综合成本
    result['annual_cost'] = 365 * result['daily_cost'] + result['inv_annual']

    # 单位供电成本
    result['unit_daily_cost'] = result['daily_cost'] / result['load_total'] if result['load_total'] > 0 else 0
    result['unit_annual_cost'] = result['annual_cost'] / (365 * result['load_total']) if result['load_total'] > 0 else 0

    # ----- 误差检查 -----
    errors = {}
    balance_errors = []
    for t in t_range:
        bal = (result['P_load_pv'][t] + result['P_load_w'][t]
               + result['P_dis'][t] + result['P_grid'][t] - load[t])
        balance_errors.append(abs(bal))
    errors['max_balance_error'] = max(balance_errors) if balance_errors else 0

    re_errors = []
    for t in t_range:
        re = (result['P_load_pv'][t] + result['P_ch_pv'][t]
              + result['P_curt_pv'][t] - G_pv[t])
        re_errors.append(abs(re))
    errors['max_re_error'] = max(re_errors) if re_errors else 0

    soc_vals = [result['E'][t] / final_E_ess if final_E_ess > 1e-6 else 0.5 for t in range(T + 1)]
    errors['min_soc'] = min(soc_vals)
    errors['max_soc'] = max(soc_vals)
    errors['end_soc_error'] = abs(result['E'][T] - result['E'][0])

    # 能量守恒
    net_energy = np.sum(ETA_C * np.array(result['P_ch']) * DT
                        - np.array(result['P_dis']) * DT / ETA_D)
    errors['net_energy_error'] = abs(net_energy)

    result['errors'] = errors

    return result


def run_no_storage(load, G_pv, G_w, curt_penalty=0.0):
    """无储能方案 - 也走 MILP 以保证成本口径一致"""
    return solve_park_optimization(load, G_pv, G_w, P_ess=0, E_ess=0,
                                   optimize_capacity=False,
                                   curt_penalty=curt_penalty)


def run_fixed_storage(load, G_pv, G_w, P_ess=50, E_ess=100, curt_penalty=0.0):
    """固定容量储能方案"""
    return solve_park_optimization(load, G_pv, G_w, P_ess=P_ess, E_ess=E_ess,
                                   optimize_capacity=False,
                                   curt_penalty=curt_penalty)


def run_optimized_storage(load, G_pv, G_w, curt_penalty=0.0):
    """连续容量理论最优方案"""
    return solve_park_optimization(load, G_pv, G_w, P_ess=0, E_ess=0,
                                   optimize_capacity=True,
                                   curt_penalty=curt_penalty)


def run_engineering_storage(
    load, G_pv, G_w,
    P_step=5, E_step=10,
    P_max=200, E_max=600,
    curt_penalty=0.0,
):
    """按工程步长直接求整数容量最优方案。"""
    return solve_park_optimization(
        load, G_pv, G_w,
        optimize_capacity=True,
        capacity_steps=(P_step, E_step),
        capacity_bounds=(P_max, E_max),
        curt_penalty=curt_penalty,
    )


def run_fixed_capacity(load, G_pv, G_w, P_ess, E_ess, curt_penalty=0.0):
    """求解指定容量下的最优运行策略（用于网格搜索/取整验证）"""
    return solve_park_optimization(load, G_pv, G_w, P_ess=P_ess, E_ess=E_ess,
                                   optimize_capacity=False,
                                   curt_penalty=curt_penalty)


if __name__ == '__main__':
    from data_loader import load_load_data, load_solar_wind_data
    import time

    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    park = 'A'
    print(f'\n===== 园区 {park} =====')
    L = load_data[park]
    P = G_pv_data[park]
    W = G_w_data[park]

    # 无储能
    print('--- 无储能 ---')
    t0 = time.time()
    r0 = run_no_storage(L, P, W)
    t1 = time.time()
    print(f'求解时间: {t1-t0:.2f}s, 状态: {r0["status"]}')
    if r0['success']:
        print(f'日运行成本: {r0["daily_cost"]:.2f} 元')
        print(f'购电量: {r0["grid_total"]:.2f} kWh')
        print(f'弃电量: {r0["curt_total"]:.2f} kWh')
        print(f'消纳率: {r0["re_ratio"]*100:.1f}%')

    # 固定储能
    print('\n--- 固定 50kW/100kWh 储能 ---')
    r1 = run_fixed_storage(L, P, W)
    if r1['success']:
        print(f'日运行成本: {r1["daily_cost"]:.2f} 元')
        print(f'年综合成本: {r1["annual_cost"]:.2f} 元')
        print(f'购电量: {r1["grid_total"]:.2f} kWh')
        print(f'弃电量: {r1["curt_total"]:.2f} kWh')
        print(f'消纳率: {r1["re_ratio"]*100:.1f}%')

    # 优化容量
    print('\n--- 优化容量 ---')
    t0 = time.time()
    r2 = run_optimized_storage(L, P, W)
    t1 = time.time()
    print(f'求解时间: {t1-t0:.2f}s, 状态: {r2["status"]}')
    if r2['success']:
        print(f'最优功率: {r2["P_ess"]:.1f} kW')
        print(f'最优容量: {r2["E_ess"]:.1f} kWh')
        print(f'年综合成本: {r2["annual_cost"]:.2f} 元')
        print(f'日运行成本: {r2["daily_cost"]:.2f} 元')

    print('\nmilp_model demo 运行完毕')
