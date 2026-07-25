# -*- coding: utf-8 -*-
"""
问题3（1）：风光储协调配置 MILP 模型

负荷增长 50%（L'=1.5L）；风光新增装机 ΔK 与储能 P/E 同时为决策变量。
- 独立运营（口径 Y）：三园区同 MILP，运行各自平衡、无园区间电力交换，
  R0 作用于汇总 R_load^ind。配置层联合决策不违背"运营独立"（Q1 之所以能分别，
  只因风光固定且无 R0；Q3 有汇总 R0 必须联合决策）。
- 联合运营：负荷与风光出力逐时聚合，共享一套储能，R0 作用于汇总 R_load^J。
- 绿色指标：风光负荷占比 R_load（非 R_re）。装机为决策变量后 R_re 可被"少建风光"
  操纵，故弃用；理由见 docs/adr/0001-rload-not-rre-when-capacity-variable.md。
- 成本口径同 Q2：光伏 0.4 / 风电 0.5 / 电网 1 元/kWh；储能充电计消纳成本；弃电免费；无 λ。
- 投资：风电 3000 / 光伏 2500 元/kW 按 5 年；储能 800P+1800E 按 10 年年化。
- 禁电网充电 + 充放电互斥：用一个二元 y_t（y=1 仅充电、y=0 放电+购电）实现三约束，
  储能充电只能用当小时风光余电（由 P_grid=0 时的功率平衡涌现）。

运行：
    $env:PYTHONUTF8='1'; & <math-python> Q3/q3_model.py        # smoke test
    & <math-python> Q3/q3_pareto.py                              # 完整 Pareto
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'templates'))
for _d in (_Q1_DIR, _TEMPLATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import (  # noqa: E402
    load_load_data, load_solar_wind_data, PARKS,
    C_PV, C_W, C_G, C_P_ESS, C_E_ESS, ETA_C, ETA_D, S_MIN, S_MAX, S_0, DT,
)

try:
    import pulp
    _HAS_PULP = True
except ImportError:
    _HAS_PULP = False

# =====================================================================
# Q3 常量
# =====================================================================
T = 24
LOAD_GROWTH = 1.5          # 最大负荷增长倍数
C_PV_INV = 2500.0          # 光伏配置成本 元/kW
C_W_INV = 3000.0           # 风电配置成本 元/kW
Y_GEN = 5                  # 风光投资回报期 年
Y_ESS = 10                 # 储能寿命 年（= data_loader.Y）
# 容量上界（grilling 共识：风光 4000；储能口径甲）
DK_MAX = 4000              # 风光新增装机上界 per 园区 per 类型 kW
P_MAX_IND, E_MAX_IND = 2000, 8000    # 独立：每园区储能上界（Q5 放大，消除 max_ind 截断）
P_MAX_JNT, E_MAX_JNT = 6000, 24000   # 联合：共享储能上界（= 三独立之和，3倍对等）
# 工程粒度
DK_STEP, P_STEP, E_STEP = 10, 5, 10

PARK_LIST = list(PARKS)    # ['A','B','C']


# =====================================================================
# 数据准备
# =====================================================================
def prepare_data():
    """读取附件1/2，返回 1.5× 负荷与归一化风光出力 φ（φ = G/K0）。"""
    load = load_load_data()
    Gpv, Gw = load_solar_wind_data()
    load15 = {p: LOAD_GROWTH * load[p] for p in PARK_LIST}
    phi_pv, phi_w = {}, {}
    for p in PARK_LIST:
        phi_pv[p] = Gpv[p] / PARKS[p]['pv'] if PARKS[p]['pv'] > 0 else np.zeros(T)
        phi_w[p] = Gw[p] / PARKS[p]['w'] if PARKS[p]['w'] > 0 else np.zeros(T)
    return {
        'load': load15,
        'phi_pv': phi_pv,
        'phi_w': phi_w,
        'K0_pv': {p: PARKS[p]['pv'] for p in PARK_LIST},
        'K0_w': {p: PARKS[p]['w'] for p in PARK_LIST},
    }


def _cap_var(name, integer, step, upper):
    """容量变量：连续 -> LpVariable；工程整数 -> step·n (n 整数)。返回表达式。"""
    if integer:
        n = pulp.LpVariable('n_' + name, lowBound=0,
                            upBound=int(upper // step), cat=pulp.LpInteger)
        return step * n
    return pulp.LpVariable(name, lowBound=0, upBound=upper)


def _val(v):
    if isinstance(v, (pulp.LpVariable, pulp.LpAffineExpression)):
        return float(pulp.value(v) or 0.0)
    return float(v)


# =====================================================================
# 独立运营（口径 Y）
# =====================================================================
def solve_independent(data, R0=None, mode='cost', integer=False, verbose=False):
    """三园区同 MILP，运行各自平衡，R0 作用于汇总 R_load^ind。

    mode='cost'      : min 年实际成本；R0 给定时加 R_load≥R0 约束，R0=None 纯经济最优。
    mode='max_rload' : max 汇总 R_load（求 R_load^max 端点，不加 R0 约束、不计成本目标）。
    integer          : True 工程整数 (ΔK=10n, P=5n, E=10n)，False 连续。
    """
    if not _HAS_PULP:
        return {'success': False, 'status': 'PuLP 未安装'}
    load = data['load']; phi_pv = data['phi_pv']; phi_w = data['phi_w']
    K0pv = data['K0_pv']; K0w = data['K0_w']

    prob = pulp.LpProblem("Q3_independent", pulp.LpMinimize)

    # ----- 容量变量 -----
    dK_pv = {p: _cap_var('dK_pv_' + p, integer, DK_STEP, DK_MAX) for p in PARK_LIST if K0pv[p] > 0}
    dK_w = {p: _cap_var('dK_w_' + p, integer, DK_STEP, DK_MAX) for p in PARK_LIST if K0w[p] > 0}
    P_ess = {p: _cap_var('P_ess_' + p, integer, P_STEP, P_MAX_IND) for p in PARK_LIST}
    E_ess = {p: _cap_var('E_ess_' + p, integer, E_STEP, E_MAX_IND) for p in PARK_LIST}

    # ----- 逐时段变量 -----
    v = {}
    for p in PARK_LIST:
        v[p] = {
            'load_pv': [pulp.LpVariable(f'Lpv_{p}_{t}', lowBound=0) for t in range(T)],
            'load_w':  [pulp.LpVariable(f'Lw_{p}_{t}', lowBound=0) for t in range(T)],
            'ch_pv':   [pulp.LpVariable(f'Cpv_{p}_{t}', lowBound=0) for t in range(T)],
            'ch_w':    [pulp.LpVariable(f'Cw_{p}_{t}', lowBound=0) for t in range(T)],
            'curt_pv': [pulp.LpVariable(f'Xpv_{p}_{t}', lowBound=0) for t in range(T)],
            'curt_w':  [pulp.LpVariable(f'Xw_{p}_{t}', lowBound=0) for t in range(T)],
            'ch':      [pulp.LpVariable(f'Ch_{p}_{t}', lowBound=0) for t in range(T)],
            'dis':     [pulp.LpVariable(f'Di_{p}_{t}', lowBound=0) for t in range(T)],
            'E':       [pulp.LpVariable(f'E_{p}_{t}', lowBound=0) for t in range(T + 1)],
            'y':       [pulp.LpVariable(f'y_{p}_{t}', cat=pulp.LpBinary) for t in range(T)],
            'grid':    [pulp.LpVariable(f'G_{p}_{t}', lowBound=0) for t in range(T)],
        }

    # ----- 约束 -----
    for p in PARK_LIST:
        has_pv = K0pv[p] > 0
        has_w = K0w[p] > 0
        peak_g = 0.0
        if has_pv:
            peak_g = max(peak_g, (K0pv[p] + DK_MAX) * float(np.max(phi_pv[p])))
        if has_w:
            peak_g = max(peak_g, (K0w[p] + DK_MAX) * float(np.max(phi_w[p])))
        Mp = max(float(np.max(load[p])), peak_g, 3000.0)
        vp = v[p]
        for t in range(T):
            if has_pv:
                Gpvt = (K0pv[p] + dK_pv[p]) * float(phi_pv[p][t])
                prob += vp['load_pv'][t] + vp['ch_pv'][t] + vp['curt_pv'][t] == Gpvt, f'pvbal_{p}_{t}'
            else:
                prob += vp['load_pv'][t] == 0, f'pvbal_{p}_{t}'
                prob += vp['ch_pv'][t] == 0, f'pvch0_{p}_{t}'
                prob += vp['curt_pv'][t] == 0, f'pvcurt0_{p}_{t}'
            if has_w:
                Gwt = (K0w[p] + dK_w[p]) * float(phi_w[p][t])
                prob += vp['load_w'][t] + vp['ch_w'][t] + vp['curt_w'][t] == Gwt, f'wbal_{p}_{t}'
            else:
                prob += vp['load_w'][t] == 0, f'wbal_{p}_{t}'
                prob += vp['ch_w'][t] == 0, f'wch0_{p}_{t}'
                prob += vp['curt_w'][t] == 0, f'wcurt0_{p}_{t}'
            prob += vp['ch'][t] == vp['ch_pv'][t] + vp['ch_w'][t], f'chsum_{p}_{t}'
            prob += (vp['load_pv'][t] + vp['load_w'][t] + vp['dis'][t] + vp['grid'][t]
                     == float(load[p][t])), f'bal_{p}_{t}'
            prob += vp['E'][t + 1] == vp['E'][t] + ETA_C * vp['ch'][t] * DT - vp['dis'][t] * DT / ETA_D, f'soc_{p}_{t}'
            prob += vp['E'][t] >= S_MIN * E_ess[p], f'socl_{p}_{t}'
            prob += vp['E'][t] <= S_MAX * E_ess[p], f'soch_{p}_{t}'
            prob += vp['ch'][t] <= P_ess[p], f'chmax_{p}_{t}'
            prob += vp['dis'][t] <= P_ess[p], f'dismax_{p}_{t}'
            # y=1 仅充电；y=0 放电+购电（充放电互斥 + 禁电网充电）
            prob += vp['ch'][t] <= Mp * vp['y'][t], f'mx1_{p}_{t}'
            prob += vp['dis'][t] <= Mp * (1 - vp['y'][t]), f'mx2_{p}_{t}'
            prob += vp['grid'][t] <= Mp * (1 - vp['y'][t]), f'mx3_{p}_{t}'
        prob += vp['E'][0] == S_0 * E_ess[p], f'init_{p}'
        prob += vp['E'][T] == vp['E'][0], f'end_{p}'

    # ----- 目标与 R0 约束 -----
    load_total = float(sum(np.sum(load[p]) for p in PARK_LIST)) * DT
    rload_num = pulp.lpSum([
        v[p]['load_pv'][t] + v[p]['load_w'][t] + v[p]['dis'][t]
        for p in PARK_LIST for t in range(T)
    ]) * DT
    daily_cost = pulp.lpSum([
        C_PV * (v[p]['load_pv'][t] + v[p]['ch_pv'][t])
        + C_W * (v[p]['load_w'][t] + v[p]['ch_w'][t])
        + C_G * v[p]['grid'][t]
        for p in PARK_LIST for t in range(T)
    ]) * DT
    inv_re = (sum(C_W_INV * dK_w[p] for p in dK_w) + sum(C_PV_INV * dK_pv[p] for p in dK_pv)) / Y_GEN
    inv_ess = sum(C_P_ESS * P_ess[p] + C_E_ESS * E_ess[p] for p in PARK_LIST) / Y_ESS
    annual = 365 * daily_cost + inv_re + inv_ess

    if mode == 'cost':
        prob += annual, 'obj'
        if R0 is not None:
            prob += rload_num >= float(R0) * load_total, 'R_load_min'
    elif mode == 'max_rload':
        prob += -rload_num, 'obj'
    else:
        raise ValueError(f"mode 必须为 'cost' 或 'max_rload'，得到 {mode}")

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=180)
    prob.solve(solver)
    return _extract_independent(prob, data, v, dK_pv, dK_w, P_ess, E_ess,
                                R0, mode, integer, load_total, annual, daily_cost,
                                inv_re, inv_ess, rload_num)


def _extract_independent(prob, data, v, dK_pv, dK_w, P_ess, E_ess,
                         R0, mode, integer, load_total, annual, daily_cost,
                         inv_re, inv_ess, rload_num):
    status = pulp.LpStatus[prob.status]
    success = prob.status == pulp.LpStatusOptimal
    res = {
        'mode': 'independent', 'R0': R0, 'solve_mode': mode, 'integer': integer,
        'success': success, 'status': status, 'objective': pulp.value(prob.objective),
    }
    if not success:
        return res
    K0pv = data['K0_pv']; K0w = data['K0_w']
    caps = {}
    for p in PARK_LIST:
        caps[p] = {
            'dK_pv': _val(dK_pv[p]) if K0pv[p] > 0 else 0.0,
            'dK_w': _val(dK_w[p]) if K0w[p] > 0 else 0.0,
            'P_ess': _val(P_ess[p]),
            'E_ess': _val(E_ess[p]),
        }
    res['capacities'] = caps
    # 派生
    dc = _val(daily_cost); ir = _val(inv_re); ie = _val(inv_ess)
    rload = _val(rload_num) / load_total
    curt_pv = sum(_val(v[p]['curt_pv'][t]) for p in PARK_LIST for t in range(T)) * DT
    curt_w = sum(_val(v[p]['curt_w'][t]) for p in PARK_LIST for t in range(T)) * DT
    grid_total = sum(_val(v[p]['grid'][t]) for p in PARK_LIST for t in range(T)) * DT
    re_gen = sum((K0pv[p] + caps[p]['dK_pv']) * float(data['phi_pv'][p][t])
                 for p in PARK_LIST for t in range(T)) * DT \
        + sum((K0w[p] + caps[p]['dK_w']) * float(data['phi_w'][p][t])
              for p in PARK_LIST for t in range(T)) * DT
    res.update({
        'daily_cost': dc, 'inv_re_ann': ir, 'inv_ess_ann': ie, 'inv_ann': ir + ie,
        'annual_cost': 365 * dc + ir + ie,
        'R_load': rload,
        'R_re': 1 - (curt_pv + curt_w) / re_gen if re_gen > 0 else 0.0,
        'load_total': load_total,
        'grid_total': grid_total, 'curt_pv': curt_pv, 'curt_w': curt_w,
        'curt_total': curt_pv + curt_w, 're_gen': re_gen,
        'hourly': {p: {k: [_val(vv) for vv in v[p][k]] for k in v[p]} for p in PARK_LIST},
    })
    res['bound_touch'] = _bound_touch_indep(caps, K0pv, K0w)
    res['errors'] = _errors_independent(data, res)
    return res


def _bound_touch_indep(caps, K0pv, K0w):
    touch = []
    for p in PARK_LIST:
        if K0pv[p] > 0 and abs(caps[p]['dK_pv'] - DK_MAX) < 1e-3:
            touch.append(f'{p}.dK_pv={DK_MAX}')
        if K0w[p] > 0 and abs(caps[p]['dK_w'] - DK_MAX) < 1e-3:
            touch.append(f'{p}.dK_w={DK_MAX}')
        if abs(caps[p]['P_ess'] - P_MAX_IND) < 1e-3:
            touch.append(f'{p}.P_ess={P_MAX_IND}')
        if abs(caps[p]['E_ess'] - E_MAX_IND) < 1e-3:
            touch.append(f'{p}.E_ess={E_MAX_IND}')
    return touch


def _errors_independent(data, res):
    load = data['load']
    errs = {'max_balance': 0.0, 'max_re_split': 0.0, 'soc_range_ok': True, 'end_soc_err': 0.0}
    for p in PARK_LIST:
        h = res['hourly'][p]
        Ecap = res['capacities'][p]['E_ess']
        for t in range(T):
            errs['max_balance'] = max(errs['max_balance'],
                abs(h['load_pv'][t] + h['load_w'][t] + h['dis'][t] + h['grid'][t] - load[p][t]))
            gpv = (res['capacities'][p]['dK_pv'] + data['K0_pv'][p]) * float(data['phi_pv'][p][t]) if data['K0_pv'][p] > 0 else 0
            gw = (res['capacities'][p]['dK_w'] + data['K0_w'][p]) * float(data['phi_w'][p][t]) if data['K0_w'][p] > 0 else 0
            errs['max_re_split'] = max(errs['max_re_split'],
                abs(h['load_pv'][t] + h['ch_pv'][t] + h['curt_pv'][t] - gpv),
                abs(h['load_w'][t] + h['ch_w'][t] + h['curt_w'][t] - gw))
        if Ecap > 1e-6:
            soc = [h['E'][t] / Ecap for t in range(T + 1)]
            if min(soc) < S_MIN - 1e-4 or max(soc) > S_MAX + 1e-4:
                errs['soc_range_ok'] = False
        errs['end_soc_err'] = max(errs['end_soc_err'], abs(h['E'][T] - h['E'][0]))
    return errs


# =====================================================================
# 联合运营
# =====================================================================
def solve_joint(data, R0=None, mode='cost', integer=False, verbose=False):
    """负荷与风光出力逐时聚合，共享一套储能，R0 作用于汇总 R_load^J。"""
    if not _HAS_PULP:
        return {'success': False, 'status': 'PuLP 未安装'}
    load = data['load']; phi_pv = data['phi_pv']; phi_w = data['phi_w']
    K0pv = data['K0_pv']; K0w = data['K0_w']
    parks_pv = [p for p in PARK_LIST if K0pv[p] > 0]
    parks_w = [p for p in PARK_LIST if K0w[p] > 0]

    prob = pulp.LpProblem("Q3_joint", pulp.LpMinimize)

    dK_pv = {p: _cap_var('dK_pv_' + p, integer, DK_STEP, DK_MAX) for p in parks_pv}
    dK_w = {p: _cap_var('dK_w_' + p, integer, DK_STEP, DK_MAX) for p in parks_w}
    P_J = _cap_var('P_J', integer, P_STEP, P_MAX_JNT)
    E_J = _cap_var('E_J', integer, E_STEP, E_MAX_JNT)

    v = {
        'load_pv': [pulp.LpVariable(f'Lpv_{t}', lowBound=0) for t in range(T)],
        'load_w':  [pulp.LpVariable(f'Lw_{t}', lowBound=0) for t in range(T)],
        'ch_pv':   [pulp.LpVariable(f'Cpv_{t}', lowBound=0) for t in range(T)],
        'ch_w':    [pulp.LpVariable(f'Cw_{t}', lowBound=0) for t in range(T)],
        'curt_pv': [pulp.LpVariable(f'Xpv_{t}', lowBound=0) for t in range(T)],
        'curt_w':  [pulp.LpVariable(f'Xw_{t}', lowBound=0) for t in range(T)],
        'ch':      [pulp.LpVariable(f'Ch_{t}', lowBound=0) for t in range(T)],
        'dis':     [pulp.LpVariable(f'Di_{t}', lowBound=0) for t in range(T)],
        'E':       [pulp.LpVariable(f'E_{t}', lowBound=0) for t in range(T + 1)],
        'y':       [pulp.LpVariable(f'y_{t}', cat=pulp.LpBinary) for t in range(T)],
        'grid':    [pulp.LpVariable(f'G_{t}', lowBound=0) for t in range(T)],
    }

    L_J = np.sum([load[p] for p in PARK_LIST], axis=0)
    load_total = float(np.sum(L_J)) * DT
    peak_g = max(
        max((sum(K0pv[p] for p in parks_pv) + DK_MAX * len(parks_pv)) * float(np.max(phi_pv[p])) for p in parks_pv) if parks_pv else 0,
        max((sum(K0w[p] for p in parks_w) + DK_MAX * len(parks_w)) * float(np.max(phi_w[p])) for p in parks_w) if parks_w else 0,
    )
    Mj = max(float(np.max(L_J)), peak_g, 3000.0)

    for t in range(T):
        Gpvt = pulp.lpSum([(K0pv[p] + dK_pv[p]) * float(phi_pv[p][t]) for p in parks_pv]) if parks_pv else 0.0
        Gwt = pulp.lpSum([(K0w[p] + dK_w[p]) * float(phi_w[p][t]) for p in parks_w]) if parks_w else 0.0
        prob += v['load_pv'][t] + v['ch_pv'][t] + v['curt_pv'][t] == Gpvt, f'pvbal_{t}'
        prob += v['load_w'][t] + v['ch_w'][t] + v['curt_w'][t] == Gwt, f'wbal_{t}'
        prob += v['ch'][t] == v['ch_pv'][t] + v['ch_w'][t], f'chsum_{t}'
        prob += v['load_pv'][t] + v['load_w'][t] + v['dis'][t] + v['grid'][t] == float(L_J[t]), f'bal_{t}'
        prob += v['E'][t + 1] == v['E'][t] + ETA_C * v['ch'][t] * DT - v['dis'][t] * DT / ETA_D, f'soc_{t}'
        prob += v['E'][t] >= S_MIN * E_J, f'socl_{t}'
        prob += v['E'][t] <= S_MAX * E_J, f'soch_{t}'
        prob += v['ch'][t] <= P_J, f'chmax_{t}'
        prob += v['dis'][t] <= P_J, f'dismax_{t}'
        prob += v['ch'][t] <= Mj * v['y'][t], f'mx1_{t}'
        prob += v['dis'][t] <= Mj * (1 - v['y'][t]), f'mx2_{t}'
        prob += v['grid'][t] <= Mj * (1 - v['y'][t]), f'mx3_{t}'
    prob += v['E'][0] == S_0 * E_J, 'init'
    prob += v['E'][T] == v['E'][0], 'end'

    rload_num = pulp.lpSum([
        v['load_pv'][t] + v['load_w'][t] + v['dis'][t] for t in range(T)
    ]) * DT
    daily_cost = pulp.lpSum([
        C_PV * (v['load_pv'][t] + v['ch_pv'][t])
        + C_W * (v['load_w'][t] + v['ch_w'][t])
        + C_G * v['grid'][t]
        for t in range(T)
    ]) * DT
    inv_re = (sum(C_W_INV * dK_w[p] for p in dK_w) + sum(C_PV_INV * dK_pv[p] for p in dK_pv)) / Y_GEN
    inv_ess = (C_P_ESS * P_J + C_E_ESS * E_J) / Y_ESS
    annual = 365 * daily_cost + inv_re + inv_ess

    if mode == 'cost':
        prob += annual, 'obj'
        if R0 is not None:
            prob += rload_num >= float(R0) * load_total, 'R_load_min'
    elif mode == 'max_rload':
        prob += -rload_num, 'obj'
    else:
        raise ValueError(f"mode 必须为 'cost' 或 'max_rload'，得到 {mode}")

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=180)
    prob.solve(solver)
    return _extract_joint(prob, data, v, dK_pv, dK_w, P_J, E_J, parks_pv, parks_w,
                           L_J, R0, mode, integer, load_total, annual, daily_cost,
                           inv_re, inv_ess, rload_num)


def _extract_joint(prob, data, v, dK_pv, dK_w, P_J, E_J, parks_pv, parks_w,
                   L_J, R0, mode, integer, load_total, annual, daily_cost,
                   inv_re, inv_ess, rload_num):
    status = pulp.LpStatus[prob.status]
    success = prob.status == pulp.LpStatusOptimal
    res = {
        'mode': 'joint', 'R0': R0, 'solve_mode': mode, 'integer': integer,
        'success': success, 'status': status, 'objective': pulp.value(prob.objective),
    }
    if not success:
        return res
    K0pv = data['K0_pv']; K0w = data['K0_w']
    caps = {p: {'dK_pv': 0.0, 'dK_w': 0.0, 'P_ess': 0.0, 'E_ess': 0.0} for p in PARK_LIST}
    for p in parks_pv:
        caps[p]['dK_pv'] = _val(dK_pv[p])
    for p in parks_w:
        caps[p]['dK_w'] = _val(dK_w[p])
    PJ = _val(P_J); EJ = _val(E_J)
    for p in PARK_LIST:
        caps[p]['P_ess'] = PJ
        caps[p]['E_ess'] = EJ
    res['capacities'] = caps
    res['P_J'] = PJ; res['E_J'] = EJ
    dc = _val(daily_cost); ir = _val(inv_re); ie = _val(inv_ess)
    rload = _val(rload_num) / load_total
    curt_pv = sum(_val(v['curt_pv'][t]) for t in range(T)) * DT
    curt_w = sum(_val(v['curt_w'][t]) for t in range(T)) * DT
    grid_total = sum(_val(v['grid'][t]) for t in range(T)) * DT
    re_gen = sum((K0pv[p] + caps[p]['dK_pv']) * float(data['phi_pv'][p][t])
                 for p in parks_pv for t in range(T)) * DT \
        + sum((K0w[p] + caps[p]['dK_w']) * float(data['phi_w'][p][t])
              for p in parks_w for t in range(T)) * DT
    res.update({
        'daily_cost': dc, 'inv_re_ann': ir, 'inv_ess_ann': ie, 'inv_ann': ir + ie,
        'annual_cost': 365 * dc + ir + ie,
        'R_load': rload,
        'R_re': 1 - (curt_pv + curt_w) / re_gen if re_gen > 0 else 0.0,
        'load_total': load_total,
        'grid_total': grid_total, 'curt_pv': curt_pv, 'curt_w': curt_w,
        'curt_total': curt_pv + curt_w, 're_gen': re_gen,
        'hourly': {k: [_val(vv) for vv in v[k]] for k in v},
        'L_J': L_J,
    })
    res['bound_touch'] = _bound_touch_joint(caps, PJ, EJ, parks_pv, parks_w)
    res['errors'] = _errors_joint(data, res, parks_pv, parks_w, L_J)
    return res


def _bound_touch_joint(caps, PJ, EJ, parks_pv, parks_w):
    touch = []
    for p in parks_pv:
        if abs(caps[p]['dK_pv'] - DK_MAX) < 1e-3:
            touch.append(f'{p}.dK_pv={DK_MAX}')
    for p in parks_w:
        if abs(caps[p]['dK_w'] - DK_MAX) < 1e-3:
            touch.append(f'{p}.dK_w={DK_MAX}')
    if abs(PJ - P_MAX_JNT) < 1e-3:
        touch.append(f'P_J={P_MAX_JNT}')
    if abs(EJ - E_MAX_JNT) < 1e-3:
        touch.append(f'E_J={E_MAX_JNT}')
    return touch


def _errors_joint(data, res, parks_pv, parks_w, L_J):
    h = res['hourly']; EJ = res['E_J']
    errs = {'max_balance': 0.0, 'max_re_split': 0.0, 'soc_range_ok': True, 'end_soc_err': 0.0}
    for t in range(T):
        errs['max_balance'] = max(errs['max_balance'],
            abs(h['load_pv'][t] + h['load_w'][t] + h['dis'][t] + h['grid'][t] - L_J[t]))
        gpv = sum((res['capacities'][p]['dK_pv'] + data['K0_pv'][p]) * float(data['phi_pv'][p][t]) for p in parks_pv)
        gw = sum((res['capacities'][p]['dK_w'] + data['K0_w'][p]) * float(data['phi_w'][p][t]) for p in parks_w)
        errs['max_re_split'] = max(errs['max_re_split'],
            abs(h['load_pv'][t] + h['ch_pv'][t] + h['curt_pv'][t] - gpv),
            abs(h['load_w'][t] + h['ch_w'][t] + h['curt_w'][t] - gw))
    if EJ > 1e-6:
        soc = [h['E'][t] / EJ for t in range(T + 1)]
        if min(soc) < S_MIN - 1e-4 or max(soc) > S_MAX + 1e-4:
            errs['soc_range_ok'] = False
    errs['end_soc_err'] = abs(h['E'][T] - h['E'][0])
    return errs


# =====================================================================
# smoke test
# =====================================================================
if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    print('=' * 64)
    print('Q3(1) 模型 smoke test')
    print('=' * 64)
    data = prepare_data()
    print(f"\n1.5x 负荷: 联合峰值={sum(np.max(data['load'][p]) for p in PARK_LIST):.1f} kW, "
          f"联合日电量={sum(np.sum(data['load'][p]) for p in PARK_LIST):.1f} kWh")

    for mode_name, fn in [('独立', solve_independent), ('联合', solve_joint)]:
        print(f"\n--- {mode_name}：纯经济最优 (连续) ---")
        r = fn(data, R0=None, mode='cost', integer=False)
        if not r['success']:
            print(f"  求解失败: {r['status']}"); continue
        caps = r['capacities']
        dKpv = sum(caps[p]['dK_pv'] for p in PARK_LIST)
        dKw = sum(caps[p]['dK_w'] for p in PARK_LIST)
        Pj = r.get('P_J', sum(caps[p]['P_ess'] for p in PARK_LIST))
        Ej = r.get('E_J', sum(caps[p]['E_ess'] for p in PARK_LIST))
        print(f"  新增 光伏={dKpv:.0f}kW 风电={dKw:.0f}kW  储能 P={Pj:.1f}/E={Ej:.1f}")
        print(f"  年实际成本={r['annual_cost']:.2f}  R_load={r['R_load']:.4f}  R_re={r['R_re']:.4f}")
        print(f"  日购电={r['grid_total']:.1f}  日弃电={r['curt_total']:.1f}  触碰上界={r['bound_touch'] or '无'}")
        print(f"  误差: bal={r['errors']['max_balance']:.2e} split={r['errors']['max_re_split']:.2e} "
              f"soc_ok={r['errors']['soc_range_ok']} end_err={r['errors']['end_soc_err']:.2e}")

        print(f"  --- {mode_name}：max R_load (连续) ---")
        rm = fn(data, R0=None, mode='max_rload', integer=False)
        if rm['success']:
            print(f"  R_load^max={rm['R_load']:.4f}  (新增光伏={sum(rm['capacities'][p]['dK_pv'] for p in PARK_LIST):.0f}, "
                  f"风电={sum(rm['capacities'][p]['dK_w'] for p in PARK_LIST):.0f}, 触碰={rm['bound_touch'] or '无'})")
    print('\nsmoke test 完毕')
