# -*- coding: utf-8 -*-
"""
问题3（2）核心 MILP 模型：12个月典型日 + 分时电价 + 风光储协调配置

关键约束：
  - 12个月独立调度，容量全年固定
  - 分时电价：峰段 1.0、谷段 0.4 元/kWh
  - 禁止弃电+购电/放电同时发生（二元约束）
  - 各月 SOC 独立循环（月初=月末=0.5Eess）
  - 年成本按各月实际天数加权
  - 仅独立运营，每园区单独求解

求解器：PuLP + HiGHS (highspy)
"""
import os
import sys
import time
import pulp
import numpy as np

# 路径
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader_12m import (
    K0_PV, K0_W, DAYS_PER_MONTH, C_PV, C_W, C_P_ESS, C_E_ESS,
    Y_ESS, ETA_C, ETA_D, S_MIN, S_MAX, S_0, DT,
)

# =====================================================================
# 常量
# =====================================================================
T = 24          # 每日小时数
NM = 12         # 月份数

# 投资参数
C_PV_INV = 2500.0
C_W_INV = 3000.0
Y_GEN = 5       # 风光投资回收期（年）

# 容量上界（沿用 Q3(1)）
DK_MAX = 4000
P_MAX = 2000
E_MAX = 8000

# 工程整数粒度
DK_STEP = 10
P_STEP = 5
E_STEP = 10

# 大M
BIG_M = 20000.0

# 求解器超时（秒）
SOLVER_TIMEOUT = 120


# =====================================================================
# 工具函数
# =====================================================================
def _val(x):
    """安全提取 PuLP 变量/表达式的值。"""
    if hasattr(x, 'value'):
        v = x.value()
        return v if v is not None else 0.0
    return float(x)


def _cap_var(name, integer, step, upper):
    """创建容量变量，可选整数步长。"""
    if integer:
        n_units = int(upper / step)
        return pulp.LpVariable(name, lowBound=0, upBound=n_units, cat='Integer') * step
    else:
        return pulp.LpVariable(name, lowBound=0, upBound=upper, cat='Continuous')


def grid_price(t):
    """分时电价。"""
    return 1.0 if 7 <= t <= 21 else 0.4


# =====================================================================
# 单园区求解器
# =====================================================================

def solve_park(park_id, phi_pv, phi_w, load, mode='cost', R0=None,
               integer_cap=False, time_limit=SOLVER_TIMEOUT, verbose=False):
    """
    为单个园区求解 Q3(2) MILP。

    Args:
        park_id: 'A', 'B', or 'C'
        phi_pv:   np.ndarray[12, 24] or None — 光伏标幺值
        phi_w:    np.ndarray[12, 24] or None — 风电标幺值
        load:     np.ndarray[12, 24] — 1.5× 负荷
        mode:     'cost' (min cost w/ R0≥) or 'max_rload'
        R0:       float or None — 最小风光负荷占比（仅 mode='cost' 生效）
        integer_cap: 是否对容量变量加整数步长约束
        time_limit: 求解器超时

    Returns:
        dict: 包含 capacities, annual_costs, monthly_details, R_load, R_re, errors
    """
    has_pv = phi_pv is not None
    has_w  = phi_w is not None
    K0pv = K0_PV[park_id] if has_pv else 0
    K0w  = K0_W[park_id] if has_w else 0

    # ---- 算 M 值 ----
    max_pv = (K0pv + DK_MAX) * float(np.max(np.abs(phi_pv))) if has_pv else 0
    max_w  = (K0w + DK_MAX) * float(np.max(np.abs(phi_w))) if has_w else 0
    M_park = max(max_pv + max_w, float(np.max(load))) * 1.5 + 100
    M_park = min(M_park, BIG_M)  # cap

    # ---- 模型 ----
    prob = pulp.LpProblem(f"Q3p2_{park_id}_{mode}", pulp.LpMinimize)

    # ==== 容量变量（全年固定）====
    dK_pv = None
    dK_w  = None
    if has_pv:
        dK_pv = _cap_var('dK_pv', integer_cap, DK_STEP, DK_MAX)
    if has_w:
        dK_w = _cap_var('dK_w', integer_cap, DK_STEP, DK_MAX)

    P_ess = _cap_var('P_ess', integer_cap, P_STEP, P_MAX)
    E_ess = _cap_var('E_ess', integer_cap, E_STEP, E_MAX)

    # ==== 每月调度变量 ====
    # 用嵌套字典: vars[m][var_name][t]
    months = []
    for m in range(NM):
        mv = {}

        # 光伏分流
        if has_pv:
            for suffix in ('pv_L', 'pv_ch', 'pv_curt'):
                mv[suffix] = [
                    pulp.LpVariable(f"{suffix}_m{m}_t{t}", lowBound=0, cat='Continuous')
                    for t in range(T)
                ]

        # 风电分流
        if has_w:
            for suffix in ('w_L', 'w_ch', 'w_curt'):
                mv[suffix] = [
                    pulp.LpVariable(f"{suffix}_m{m}_t{t}", lowBound=0, cat='Continuous')
                    for t in range(T)
                ]

        # 总充电 = pv_ch + w_ch
        mv['ch'] = [
            pulp.LpVariable(f"ch_m{m}_t{t}", lowBound=0, cat='Continuous')
            for t in range(T)
        ]

        # 放电、电网购电
        mv['dis'] = [
            pulp.LpVariable(f"dis_m{m}_t{t}", lowBound=0, cat='Continuous')
            for t in range(T)
        ]
        mv['grid'] = [
            pulp.LpVariable(f"grid_m{m}_t{t}", lowBound=0, cat='Continuous')
            for t in range(T)
        ]

        # SOC (0..T 共 T+1 个点)
        mv['soc'] = [
            pulp.LpVariable(f"soc_m{m}_t{t}", lowBound=0, cat='Continuous')
            for t in range(T + 1)
        ]

        # 二元变量
        mv['y'] = [  # y=1: 仅充电; y=0: 购电/放电
            pulp.LpVariable(f"y_m{m}_t{t}", cat='Binary')
            for t in range(T)
        ]
        mv['u'] = [  # u=1: 允许弃电; u=0: 购电/放电可行
            pulp.LpVariable(f"u_m{m}_t{t}", cat='Binary')
            for t in range(T)
        ]

        months.append(mv)

    # ==== 约束 ====
    for m in range(NM):
        mv = months[m]

        # 总充电 = 各源充电之和
        ch_terms = []
        if has_pv:
            ch_terms.extend(mv['pv_ch'])
        if has_w:
            ch_terms.extend(mv['w_ch'])
        for t in range(T):
            lhs = sum(terms[t] for terms in (
                [mv['pv_ch']] if has_pv else []) +
                ([mv['w_ch']] if has_w else []))
            prob += mv['ch'][t] == lhs, f"ch_eq_m{m}_t{t}"

        # 风光发电平衡
        if has_pv:
            for t in range(T):
                Gpv = (K0pv + dK_pv) * float(phi_pv[m, t])
                prob += (mv['pv_L'][t] + mv['pv_ch'][t] + mv['pv_curt'][t]
                         == Gpv, f"pv_bal_m{m}_t{t}")

        if has_w:
            for t in range(T):
                Gw = (K0w + dK_w) * float(phi_w[m, t])
                prob += (mv['w_L'][t] + mv['w_ch'][t] + mv['w_curt'][t]
                         == Gw, f"w_bal_m{m}_t{t}")

        # 功率平衡
        for t in range(T):
            lhs = mv['dis'][t] + mv['grid'][t]
            if has_pv:
                lhs += mv['pv_L'][t]
            if has_w:
                lhs += mv['w_L'][t]
            prob += lhs == float(load[m, t]), f"pwr_bal_m{m}_t{t}"

        # SOC 递推
        for t in range(T):
            prob += (mv['soc'][t + 1]
                     == mv['soc'][t]
                     + ETA_C * mv['ch'][t] * DT
                     - mv['dis'][t] * DT / ETA_D,
                     f"soc_step_m{m}_t{t}")

        # SOC 上下限
        for t in range(T + 1):
            prob += mv['soc'][t] >= S_MIN * E_ess, f"soc_lo_m{m}_t{t}"
            prob += mv['soc'][t] <= S_MAX * E_ess, f"soc_hi_m{m}_t{t}"

        # 日初 = 日末 = 0.5*E
        prob += mv['soc'][0] == S_0 * E_ess, f"soc_start_m{m}"
        prob += mv['soc'][T] == S_0 * E_ess, f"soc_end_m{m}"

        # 充放电功率 ≤ P_ess
        for t in range(T):
            prob += mv['ch'][t] <= P_ess, f"ch_lim_m{m}_t{t}"
            prob += mv['dis'][t] <= P_ess, f"dis_lim_m{m}_t{t}"

        # 二元约束：充电 vs 购电+放电 互斥 (§6.1)
        for t in range(T):
            prob += mv['ch'][t] <= M_park * mv['y'][t], f"ch_y_m{m}_t{t}"
            prob += (mv['grid'][t] + mv['dis'][t]
                     <= M_park * (1 - mv['y'][t]), f"grid_dis_y_m{m}_t{t}")

        # 二元约束：弃电 vs 购电+放电 互斥 (§6.2)
        curt_terms = []
        if has_pv:
            curt_terms.extend(mv['pv_curt'])
        if has_w:
            curt_terms.extend(mv['w_curt'])
        for t in range(T):
            total_curt = sum(terms[t] for terms in (
                [mv['pv_curt']] if has_pv else []) +
                ([mv['w_curt']] if has_w else []))
            prob += total_curt <= M_park * mv['u'][t], f"curt_u_m{m}_t{t}"
            prob += (mv['grid'][t] + mv['dis'][t]
                     <= M_park * (1 - mv['u'][t]), f"grid_dis_u_m{m}_t{t}")

    # ==== 目标函数 ====

    # 各月典型日运行成本
    monthly_daily_cost = []
    for m in range(NM):
        mv = months[m]
        daily_cost = 0.0
        for t in range(T):
            # 风光消纳成本
            if has_pv:
                daily_cost += C_PV * (mv['pv_L'][t] + mv['pv_ch'][t])
            if has_w:
                daily_cost += C_W * (mv['w_L'][t] + mv['w_ch'][t])
            # 电网购电（分时电价）
            daily_cost += grid_price(t) * mv['grid'][t]
        monthly_daily_cost.append(daily_cost)

    # 年运行成本
    annual_running = pulp.lpSum(
        DAYS_PER_MONTH[m] * monthly_daily_cost[m] for m in range(NM)
    )

    # 年化投资成本
    inv_gen = 0.0
    if has_pv:
        inv_gen += C_PV_INV * dK_pv / Y_GEN
    if has_w:
        inv_gen += C_W_INV * dK_w / Y_GEN
    inv_ess = (C_P_ESS * P_ess + C_E_ESS * E_ess) / Y_ESS
    annual_inv = inv_gen + inv_ess

    annual_total = annual_running + annual_inv

    # R_load（年加权）
    annual_load_total = sum(
        DAYS_PER_MONTH[m] * sum(float(load[m, t]) for t in range(T))
        for m in range(NM)
    )
    annual_re_to_load = pulp.lpSum(
        DAYS_PER_MONTH[m] * (
            pulp.lpSum([mv['dis'][t] for t in range(T)] +
                       ([mv['pv_L'][t] for t in range(T)] if has_pv else []) +
                       ([mv['w_L'][t] for t in range(T)] if has_w else []))
        )
        for m, mv in enumerate(months)
    )
    R_load = annual_re_to_load / annual_load_total

    # 设置目标
    if mode == 'max_rload':
        prob += -annual_re_to_load  # max R_load ⇔ max re_to_load
    else:
        # mode == 'cost'
        if R0 is not None:
            prob += R_load >= R0, "R0_constraint"
        prob += annual_total, "min_cost"

    # ==== 求解 ====
    solver = pulp.HiGHS(
        msg=verbose,
        timeLimit=time_limit,
        options={'mip_abs_gap': 0.1, 'mip_rel_gap': 0.0001},
    )

    if verbose:
        print(f"[solve] park={park_id} mode={mode} R0={R0} int={integer_cap} ...")

    t0 = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    if verbose:
        print(f"  status={pulp.LpStatus[status]}, time={elapsed:.1f}s, "
              f"obj={_val(prob.objective):.2f}")

    # ==== 提取结果 ====
    caps = {
        'dK_pv': _val(dK_pv) if has_pv else 0.0,
        'dK_w':  _val(dK_w) if has_w else 0.0,
        'P_ess': _val(P_ess),
        'E_ess': _val(E_ess),
    }
    caps['K_pv_total'] = K0pv + caps['dK_pv']
    caps['K_w_total']  = K0w + caps['dK_w']

    # 逐月结果
    monthly = []
    for m in range(NM):
        mv = months[m]
        md = {
            'pv_L': [_val(mv['pv_L'][t]) for t in range(T)] if has_pv else [0]*T,
            'pv_ch': [_val(mv['pv_ch'][t]) for t in range(T)] if has_pv else [0]*T,
            'pv_curt': [_val(mv['pv_curt'][t]) for t in range(T)] if has_pv else [0]*T,
            'w_L': [_val(mv['w_L'][t]) for t in range(T)] if has_w else [0]*T,
            'w_ch': [_val(mv['w_ch'][t]) for t in range(T)] if has_w else [0]*T,
            'w_curt': [_val(mv['w_curt'][t]) for t in range(T)] if has_w else [0]*T,
            'ch': [_val(mv['ch'][t]) for t in range(T)],
            'dis': [_val(mv['dis'][t]) for t in range(T)],
            'grid': [_val(mv['grid'][t]) for t in range(T)],
            'soc': [_val(mv['soc'][t]) for t in range(T + 1)],
            'y': [_val(mv['y'][t]) for t in range(T)],
            'u': [_val(mv['u'][t]) for t in range(T)],
        }
        # 派生指标
        md['G_pv'] = [(K0pv + caps['dK_pv']) * float(phi_pv[m, t]) for t in range(T)] if has_pv else [0]*T
        md['G_w']  = [(K0w + caps['dK_w']) * float(phi_w[m, t]) for t in range(T)] if has_w else [0]*T

        # 日运行成本
        md['daily_cost'] = 0.0
        for t in range(T):
            if has_pv:
                md['daily_cost'] += C_PV * (md['pv_L'][t] + md['pv_ch'][t])
            if has_w:
                md['daily_cost'] += C_W * (md['w_L'][t] + md['w_ch'][t])
            md['daily_cost'] += grid_price(t) * md['grid'][t]

        # 日购电量
        md['grid_total'] = sum(md['grid'])
        # 日弃电量
        md['curt_total'] = sum(md['pv_curt']) + sum(md['w_curt'])
        # 日风光供负荷（直接+经储能放电）
        md['re_to_load'] = sum(md['pv_L']) + sum(md['w_L']) + sum(md['dis'])
        # 日总负荷
        md['load_total'] = sum(float(load[m, t]) for t in range(T))
        # 风光总出力
        md['gen_total'] = sum(md['G_pv']) + sum(md['G_w'])

        monthly.append(md)

    # 年汇总
    annual = {}
    annual['running_cost'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['daily_cost'] for m in range(NM)
    )
    annual['inv_gen'] = _val(inv_gen)
    annual['inv_ess'] = _val(inv_ess)
    annual['inv_total'] = _val(annual_inv)
    annual['total_cost'] = annual['running_cost'] + annual['inv_total']
    annual['grid_total'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['grid_total'] for m in range(NM)
    )
    annual['curt_total'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['curt_total'] for m in range(NM)
    )
    annual['re_to_load'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['re_to_load'] for m in range(NM)
    )
    annual['load_total'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['load_total'] for m in range(NM)
    )
    annual['gen_total'] = sum(
        DAYS_PER_MONTH[m] * monthly[m]['gen_total'] for m in range(NM)
    )
    annual['R_load'] = (annual['re_to_load'] / annual['load_total']
                        if annual['load_total'] > 0 else 0.0)
    annual['R_re'] = (1.0 - annual['curt_total'] / annual['gen_total']
                      if annual['gen_total'] > 0 else 0.0)

    # 分峰谷购电统计
    peak_grid = 0.0
    valley_grid = 0.0
    peak_cost = 0.0
    valley_cost = 0.0
    for m in range(NM):
        for t in range(T):
            g = monthly[m]['grid'][t]
            if 7 <= t <= 21:
                peak_grid += DAYS_PER_MONTH[m] * g
                peak_cost += DAYS_PER_MONTH[m] * g * 1.0
            else:
                valley_grid += DAYS_PER_MONTH[m] * g
                valley_cost += DAYS_PER_MONTH[m] * g * 0.4
    annual['peak_grid'] = peak_grid
    annual['valley_grid'] = valley_grid
    annual['peak_cost'] = peak_cost
    annual['valley_cost'] = valley_cost

    # 约束误差检查
    errors = _check_errors(park_id, caps, monthly, load, phi_pv, phi_w)

    return {
        'park': park_id,
        'mode': mode,
        'R0': R0,
        'integer': integer_cap,
        'status': pulp.LpStatus[status],
        'solve_time': elapsed,
        'capacities': caps,
        'annual': annual,
        'monthly': monthly,
        'errors': errors,
        'bound_touch': _check_bounds(caps),
    }


# =====================================================================
# 结果检查
# =====================================================================
def _check_errors(park_id, caps, monthly, load, phi_pv, phi_w):
    """验证约束满足情况。"""
    errs = []
    tol = 1.0  # kW level tolerance

    Kpv = caps['K_pv_total']
    Kw = caps['K_w_total']
    P_ess = caps['P_ess']
    E_ess = caps['E_ess']
    has_pv = Kpv > 0
    has_w = Kw > 0

    for m in range(NM):
        md = monthly[m]
        # --- 功率平衡 ---
        for t in range(T):
            lhs = md['dis'][t] + md['grid'][t]
            if has_pv:
                lhs += md['pv_L'][t]
            if has_w:
                lhs += md['w_L'][t]
            rhs = float(load[m, t])
            if abs(lhs - rhs) > tol:
                errs.append(f"M{m}T{t}: 功率不平衡 {lhs:.1f}≠{rhs:.1f}")

        # --- 光伏平衡 ---
        if has_pv:
            for t in range(T):
                Gpv = Kpv * float(phi_pv[m, t])
                bal = md['pv_L'][t] + md['pv_ch'][t] + md['pv_curt'][t]
                if abs(bal - Gpv) > tol:
                    errs.append(f"M{m}T{t}: 光伏不平衡 {bal:.1f}≠{Gpv:.1f}")

        # --- 风电平衡 ---
        if has_w:
            for t in range(T):
                Gw = Kw * float(phi_w[m, t])
                bal = md['w_L'][t] + md['w_ch'][t] + md['w_curt'][t]
                if abs(bal - Gw) > tol:
                    errs.append(f"M{m}T{t}: 风电不平衡 {bal:.1f}≠{Gw:.1f}")

        # --- SOC 递推 ---
        for t in range(T):
            expected = md['soc'][t] + ETA_C * md['ch'][t] * DT - md['dis'][t] * DT / ETA_D
            if abs(md['soc'][t + 1] - expected) > tol:
                errs.append(f"M{m}T{t}: SOC步进 {md['soc'][t+1]:.1f}≠{expected:.1f}")

        # --- SOC 端点 ---
        if abs(md['soc'][0] - S_0 * E_ess) > tol:
            errs.append(f"M{m}: SOC初值 {md['soc'][0]:.1f}≠{S_0*E_ess:.1f}")
        if abs(md['soc'][T] - S_0 * E_ess) > tol:
            errs.append(f"M{m}: SOC末值 {md['soc'][T]:.1f}≠{S_0*E_ess:.1f}")

        # --- SOC 范围 ---
        for t in range(T + 1):
            if md['soc'][t] < S_MIN * E_ess - tol:
                errs.append(f"M{m}T{t}: SOC低于下限 {md['soc'][t]:.1f}<{S_MIN*E_ess:.1f}")
            if md['soc'][t] > S_MAX * E_ess + tol:
                errs.append(f"M{m}T{t}: SOC高于上限 {md['soc'][t]:.1f}>{S_MAX*E_ess:.1f}")

        # --- 充放电互斥 ---
        for t in range(T):
            if md['ch'][t] > 0.01 and md['dis'][t] > 0.01:
                errs.append(f"M{m}T{t}: 同时充放电 ch={md['ch'][t]:.1f} dis={md['dis'][t]:.1f}")

        # --- 购电+弃电互斥 ---
        for t in range(T):
            curt = md['pv_curt'][t] + md['w_curt'][t]
            if curt > 0.01 and md['grid'][t] > 0.01:
                errs.append(f"M{m}T{t}: 同时购电+弃电 grid={md['grid'][t]:.1f} curt={curt:.1f}")

        # --- 购电+充电互斥（电网不能给储能充电）---
        for t in range(T):
            if md['ch'][t] > 0.01 and md['grid'][t] > 0.01:
                errs.append(f"M{m}T{t}: 购电同时充电 grid={md['grid'][t]:.1f} ch={md['ch'][t]:.1f}")

        # --- 充放电功率 ---
        for t in range(T):
            if md['ch'][t] > P_ess + tol:
                errs.append(f"M{m}T{t}: 充电>{P_ess} ch={md['ch'][t]:.1f}")
            if md['dis'][t] > P_ess + tol:
                errs.append(f"M{m}T{t}: 放电>{P_ess} dis={md['dis'][t]:.1f}")

    return errs


def _check_bounds(caps):
    """检查容量是否触碰上界。"""
    touches = []
    if abs(caps['dK_pv'] - DK_MAX) < 1e-3 and caps['dK_pv'] > 0:
        touches.append('dK_pv')
    if abs(caps['dK_w'] - DK_MAX) < 1e-3 and caps['dK_w'] > 0:
        touches.append('dK_w')
    if abs(caps['P_ess'] - P_MAX) < 1e-3:
        touches.append('P_ess')
    if abs(caps['E_ess'] - E_MAX) < 1e-3:
        touches.append('E_ess')
    return touches


# =====================================================================
# Smoke test
# =====================================================================
if __name__ == '__main__':
    from data_loader_12m import load_12m_data, load_load_12m

    print("=" * 60)
    print("Q3(2) 模型 smoke test")
    print("=" * 60)

    data = load_12m_data()
    loads = load_load_12m()

    # 测试园区 A：纯经济最优（连续容量）
    print("\n--- Park A: 纯经济最优 (continuous) ---")
    r = solve_park('A', data['phi_pv']['A'], None, loads['A'],
                   mode='cost', R0=None, integer_cap=False, verbose=True)
    print(f"  ΔKpv={r['capacities']['dK_pv']:.0f}kW  "
          f"P={r['capacities']['P_ess']:.0f}kW  "
          f"E={r['capacities']['E_ess']:.0f}kWh")
    print(f"  年总成本={r['annual']['total_cost']:.2f}元  "
          f"R_load={r['annual']['R_load']:.4f}  "
          f"R_re={r['annual']['R_re']:.4f}")
    print(f"  误差数: {len(r['errors'])}")
    if r['errors']:
        for e in r['errors'][:5]:
            print(f"    !! {e}")

    # 测试园区 B：纯经济最优（连续容量）
    print("\n--- Park B: 纯经济最优 (continuous) ---")
    r2 = solve_park('B', None, data['phi_w']['B'], loads['B'],
                    mode='cost', R0=None, integer_cap=False, verbose=True)
    print(f"  ΔKw={r2['capacities']['dK_w']:.0f}kW  "
          f"P={r2['capacities']['P_ess']:.0f}kW  "
          f"E={r2['capacities']['E_ess']:.0f}kWh")
    print(f"  年总成本={r2['annual']['total_cost']:.2f}元  "
          f"R_load={r2['annual']['R_load']:.4f}  "
          f"R_re={r2['annual']['R_re']:.4f}")
    print(f"  误差数: {len(r2['errors'])}")

    # 测试园区 C：纯经济最优（连续容量）
    print("\n--- Park C: 纯经济最优 (continuous) ---")
    r3 = solve_park('C', data['phi_pv']['C'], data['phi_w']['C'], loads['C'],
                    mode='cost', R0=None, integer_cap=False, verbose=True)
    print(f"  ΔKpv={r3['capacities']['dK_pv']:.0f}kW  "
          f"ΔKw={r3['capacities']['dK_w']:.0f}kW  "
          f"P={r3['capacities']['P_ess']:.0f}kW  "
          f"E={r3['capacities']['E_ess']:.0f}kWh")
    print(f"  年总成本={r3['annual']['total_cost']:.2f}元  "
          f"R_load={r3['annual']['R_load']:.4f}  "
          f"R_re={r3['annual']['R_re']:.4f}")
    print(f"  误差数: {len(r3['errors'])}")
    if r3['errors']:
        for e in r3['errors'][:5]:
            print(f"    !! {e}")

    print("\n" + "=" * 60)
    print("Smoke test 完成")
