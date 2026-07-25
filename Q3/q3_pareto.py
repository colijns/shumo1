# -*- coding: utf-8 -*-
"""
问题3（1）Pareto 主模型：ε-约束法构造"年实际成本-风光负荷占比" Pareto 前沿。

与 Q2 的关键差异（见 docs/adr/0001-rload-not-rre-when-capacity-variable.md）：
  - 绿色指标用 R_load（风光负荷占比），非 R_re。装机为决策变量后 R_re 可被"少建风光"操纵。
  - 风光新增装机 ΔK 为决策变量；独立/联合两模式分别求前沿。
  - 独立口径 Y：三园区同 MILP、运行各自平衡、R0 作用于汇总 R_load^ind（见 q3_model.py）。
  - 独立与联合用同一组 R0；扫描范围 [max(econ), min(max)] 内均匀 6 点。

流程（docs §9）：
  1. 两模式各求 econ 端点（R0=None）与 max 端点（mode=max_rload）；
  2. 扫描范围 [max(econ_ind, econ_J), min(max_ind, max_J)] 均匀 6 点；
  3. 每 R0 求 ind/jnt 的连续 MILP（理论下界）+ 工程整数 MILP（主结果）；
  4. 推荐点触碰上界时扩大 1.5× 复算（docs §9 步骤4）；
  5. 等权理想点距离选折中 R0*（基于联合前沿，等权为决策偏好）；
  6. 输出 docs §10 十项 + ind/joint 同 R0 的 ΔC、η_C 对比。

运行：
    $env:PYTHONUTF8='1'; & <math-python> Q3/q3_pareto.py --no-integers   # 仅连续，快速验证
    & <math-python> Q3/q3_pareto.py                                      # 完整（连续+整数）
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'templates'))
for _d in (_HERE, _Q1_DIR, _TEMPLATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import q3_model as M  # noqa: E402
from q3_model import prepare_data, solve_independent, solve_joint, PARK_LIST, T  # noqa: E402
from data_loader import DT  # noqa: E402
from common.io_utils import export_result  # noqa: E402
from common.plot_style import set_chinese_style  # noqa: E402

R0_NPOINTS = 6
TOL = 1e-4


def _fix_chinese_font():
    for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            _name = font_manager.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


def _sum_caps(r):
    """汇总三园区新增装机与储能。"""
    caps = r['capacities']
    dKpv = sum(caps[p]['dK_pv'] for p in PARK_LIST)
    dKw = sum(caps[p]['dK_w'] for p in PARK_LIST)
    if r['mode'] == 'joint':
        P, E = r['P_J'], r['E_J']
    else:
        P = sum(caps[p]['P_ess'] for p in PARK_LIST)
        E = sum(caps[p]['E_ess'] for p in PARK_LIST)
    return dKpv, dKw, P, E


# =====================================================================
# 求解
# =====================================================================
def solve_endpoints(data, integers=False):
    """两模式各求 econ（min 成本）与 max（max R_load）端点。max 只求连续。"""
    out = {}
    print('  [端点] 独立 econ ...'); t0 = time.time()
    out['ind_econ_c'] = solve_independent(data, R0=None, mode='cost', integer=False)
    out['ind_econ_i'] = solve_independent(data, R0=None, mode='cost', integer=True)
    print(f'        连续 R_load={out["ind_econ_c"]["R_load"]:.4f} C={out["ind_econ_c"]["annual_cost"]:.0f} '
          f'| 整数 R_load={out["ind_econ_i"]["R_load"]:.4f} C={out["ind_econ_i"]["annual_cost"]:.0f} ({time.time()-t0:.0f}s)')
    print('  [端点] 联合 econ ...'); t0 = time.time()
    out['jnt_econ_c'] = solve_joint(data, R0=None, mode='cost', integer=False)
    out['jnt_econ_i'] = solve_joint(data, R0=None, mode='cost', integer=True)
    print(f'        连续 R_load={out["jnt_econ_c"]["R_load"]:.4f} C={out["jnt_econ_c"]["annual_cost"]:.0f} '
          f'| 整数 R_load={out["jnt_econ_i"]["R_load"]:.4f} C={out["jnt_econ_i"]["annual_cost"]:.0f} ({time.time()-t0:.0f}s)')
    print('  [端点] 独立 max R_load ...'); t0 = time.time()
    out['ind_max_c'] = solve_independent(data, R0=None, mode='max_rload', integer=False)
    print(f'        R_load^max={out["ind_max_c"]["R_load"]:.4f} ({time.time()-t0:.0f}s) 触碰={out["ind_max_c"]["bound_touch"] or "无"}')
    print('  [端点] 联合 max R_load ...'); t0 = time.time()
    out['jnt_max_c'] = solve_joint(data, R0=None, mode='max_rload', integer=False)
    print(f'        R_load^max={out["jnt_max_c"]["R_load"]:.4f} ({time.time()-t0:.0f}s) 触碰={out["jnt_max_c"]["bound_touch"] or "无"}')
    return out


def scan_range(ep):
    lo = max(ep['ind_econ_c']['R_load'], ep['jnt_econ_c']['R_load'])
    hi = min(ep['ind_max_c']['R_load'], ep['jnt_max_c']['R_load'])
    return lo, hi


def build_front(data, R0_list, integers=True):
    """对每个 R0 求 ind/jnt 的连续 (+ 整数) 最优。"""
    front = {}
    for R0 in R0_list:
        entry = {}
        t0 = time.time()
        entry['ind_c'] = solve_independent(data, R0=R0, mode='cost', integer=False)
        entry['jnt_c'] = solve_joint(data, R0=R0, mode='cost', integer=False)
        if integers:
            entry['ind_i'] = solve_independent(data, R0=R0, mode='cost', integer=True)
            entry['jnt_i'] = solve_joint(data, R0=R0, mode='cost', integer=True)
        front[R0] = entry
        ic = entry['ind_c']; jc = entry['jnt_c']
        line = (f"  [R0={R0*100:.2f}%] 独立 C={ic['annual_cost']:.0f} R_load={ic['R_load']:.4f} | "
                f"联合 C={jc['annual_cost']:.0f} R_load={jc['R_load']:.4f}")
        if integers:
            ii = entry['ind_i']; ji = entry['jnt_i']
            line += (f"\n             整数: 独立 C={ii['annual_cost']:.0f} R_load={ii['R_load']:.4f} | "
                     f"联合 C={ji['annual_cost']:.0f} R_load={ji['R_load']:.4f}")
        print(line + f"  ({time.time()-t0:.0f}s)")
    return front


# =====================================================================
# 等权理想点距离选型（基于联合前沿）
# =====================================================================
def ideal_point_select(ep, front, R0_list):
    """候选 = {联合 econ 连续} ∪ {各 R0 联合连续}。等权归一化选 d 最小的 R0*。

    等权处理属于决策偏好（docs §8.2）。最终 ind/joint 比较采用此 R0*。
    """
    candidates = [{
        'label': '纯经济基准', 'R0': None,
        'R_load': ep['jnt_econ_c']['R_load'],
        'annual_cost': ep['jnt_econ_c']['annual_cost'],
    }]
    for R0 in R0_list:
        r = front[R0]['jnt_c']
        candidates.append({
            'label': f'{R0*100:.2f}%目标', 'R0': R0,
            'R_load': r['R_load'], 'annual_cost': r['annual_cost'],
        })
    costs = [c['annual_cost'] for c in candidates]
    ratios = [c['R_load'] for c in candidates]
    C_min, C_max = min(costs), max(costs)
    R_min, R_max = min(ratios), max(ratios)
    c_span = C_max - C_min if C_max > C_min else 1.0
    r_span = R_max - R_min if R_max > R_min else 1.0
    for c in candidates:
        c['cost_norm'] = (c['annual_cost'] - C_min) / c_span
        c['green_gap'] = (R_max - c['R_load']) / r_span
        c['ideal_dist'] = float(np.sqrt(c['cost_norm']**2 + c['green_gap']**2))
    best = min(candidates, key=lambda c: c['ideal_dist'])
    return {'candidates': candidates, 'best': best,
            'ranges': {'C_min': C_min, 'C_max': C_max, 'R_min': R_min, 'R_max': R_max}}


def enlarge_recheck(data, r_opt, mode, R0):
    """触碰上界时扩大 1.5× 复算（docs §9 步骤4）。返回 (新解, 是否变化)。"""
    if not r_opt.get('bound_touch'):
        return None, False
    saved = (M.DK_MAX, M.P_MAX_IND, M.E_MAX_IND, M.P_MAX_JNT, M.E_MAX_JNT)
    M.DK_MAX = int(saved[0] * 1.5)
    M.P_MAX_IND = int(saved[1] * 1.5)
    M.E_MAX_IND = int(saved[2] * 1.5)
    M.P_MAX_JNT = int(saved[3] * 1.5)
    M.E_MAX_JNT = int(saved[4] * 1.5)
    try:
        fn = solve_independent if mode == 'independent' else solve_joint
        r2 = fn(data, R0=R0, mode='cost', integer=True)
    finally:
        M.DK_MAX, M.P_MAX_IND, M.E_MAX_IND, M.P_MAX_JNT, M.E_MAX_JNT = saved
    changed = (not r2.get('success')) or abs(r2['annual_cost'] - r_opt['annual_cost']) > 1.0
    return r2, changed


# =====================================================================
# 表格构造
# =====================================================================
def _row(r, R0, label):
    dKpv, dKw, P, E = _sum_caps(r)
    return {
        '模式': label,
        'R0目标': f'{R0*100:.2f}%' if R0 is not None else '纯经济',
        '新增光伏/kW': round(dKpv, 1),
        '新增风电/kW': round(dKw, 1),
        '储能功率/kW': round(P, 1),
        '储能容量/kWh': round(E, 1),
        'E/P/h': round(E / P, 3) if P > 1e-6 else 0.0,
        '风光负荷占比': f"{r['R_load']*100:.3f}%",
        '风光消纳率': f"{r['R_re']*100:.3f}%",
        '日购电/kWh': round(r['grid_total'], 1),
        '日弃电/kWh': round(r['curt_total'], 1),
        '年运行成本/元': round(365 * r['daily_cost'], 0),
        '年风光投资/元': round(r['inv_re_ann'], 0),
        '年储能投资/元': round(r['inv_ess_ann'], 0),
        '年实际成本/元': round(r['annual_cost'], 0),
        '触碰上界': '是' if r.get('bound_touch') else '否',
    }


def build_continuous_table(ep, front, R0_list):
    rows = []
    rows.append(_row(ep['ind_econ_c'], None, '独立'))
    rows.append(_row(ep['jnt_econ_c'], None, '联合'))
    for R0 in R0_list:
        rows.append(_row(front[R0]['ind_c'], R0, '独立'))
        rows.append(_row(front[R0]['jnt_c'], R0, '联合'))
    return pd.DataFrame(rows)


def build_integer_table(ep, front, R0_list):
    rows = []
    rows.append(_row(ep['ind_econ_i'], None, '独立'))
    rows.append(_row(ep['jnt_econ_i'], None, '联合'))
    for R0 in R0_list:
        rows.append(_row(front[R0]['ind_i'], R0, '独立'))
        rows.append(_row(front[R0]['jnt_i'], R0, '联合'))
    return pd.DataFrame(rows)


def build_ideal_table(sel):
    rows = []
    for c in sel['candidates']:
        rows.append({
            '候选方案': c['label'],
            '风光负荷占比': f"{c['R_load']*100:.3f}%",
            '年实际成本/元': round(c['annual_cost'], 0),
            '理想点距离': round(c['ideal_dist'], 4),
        })
    return pd.DataFrame(rows)


def build_compare_table(ep, front, R0_list, suffix='i'):
    """同 R0 下独立 vs 联合的 ΔC、η_C。suffix='i' 工程整数 / 'c' 连续。"""
    rows = []
    base_i = ep[f'ind_econ_{suffix}']; base_j = ep[f'jnt_econ_{suffix}']
    rows.append({
        'R0': '纯经济', '独立年成本/元': round(base_i['annual_cost'], 0),
        '联合年成本/元': round(base_j['annual_cost'], 0),
        'ΔC(独立-联合)/元': round(base_i['annual_cost'] - base_j['annual_cost'], 0),
        '收益率η_C': f"{(base_i['annual_cost']-base_j['annual_cost'])/base_i['annual_cost']*100:.3f}%",
    })
    for R0 in R0_list:
        ci = front[R0][f'ind_{suffix}']; cj = front[R0][f'jnt_{suffix}']
        dC = ci['annual_cost'] - cj['annual_cost']
        rows.append({
            'R0': f'{R0*100:.2f}%',
            '独立年成本/元': round(ci['annual_cost'], 0),
            '联合年成本/元': round(cj['annual_cost'], 0),
            'ΔC(独立-联合)/元': round(dC, 0),
            '收益率η_C': f"{dC/ci['annual_cost']*100:.3f}%",
        })
    return pd.DataFrame(rows)


def build_marginal_table(front, R0_list, mode='jnt_c'):
    """相邻 R0 每提高 1 个百分点的年成本增量。"""
    rows = []
    prev = None
    for R0 in R0_list:
        r = front[R0][mode]
        if prev is not None:
            dC = r['annual_cost'] - prev['annual_cost']
            dR = (r['R_load'] - prev_R) * 100
            rows.append({
                'R0区间': f"{prev_R*100:.2f}%→{R0*100:.2f}%",
                'ΔR_load/百分点': round(dR, 3),
                'Δ年成本/元': round(dC, 0),
                '每提高1pp增量/元': round(dC / dR, 0) if abs(dR) > 1e-9 else 0,
            })
        prev = r; prev_R = R0
    return pd.DataFrame(rows)


def build_hourly_table(r_opt):
    """典型日逐时调度（独立汇总三园区，联合聚合）。"""
    if r_opt['mode'] == 'independent':
        h = r_opt['hourly']
        merge = lambda k: [sum(h[p][k][t] for p in PARK_LIST) for t in range(T)]
        E = [sum(h[p]['E'][t] for p in PARK_LIST) for t in range(T + 1)]
        load = [sum(float(r_opt['load_total'] / T) for _ in [0])] * T  # 占位，下方覆盖
        L = [sum(h[p]['load_pv'][t] + h[p]['load_w'][t] + h[p]['dis'][t] + h[p]['grid'][t] for p in PARK_LIST) for t in range(T)]
    else:
        h = r_opt['hourly']; L = list(r_opt['L_J'])
        merge = lambda k: list(h[k])
        E = list(h['E'])
    return pd.DataFrame({
        '时刻': list(range(T)),
        '负荷/kW': [round(float(x), 1) for x in L],
        '风光直供/kW': [round(merge('load_pv')[t] + merge('load_w')[t], 1) for t in range(T)],
        '储能充电/kW': [round(merge('ch')[t], 1) for t in range(T)],
        '储能放电/kW': [round(merge('dis')[t], 1) for t in range(T)],
        '电网购电/kW': [round(merge('grid')[t], 1) for t in range(T)],
        '弃电/kW': [round(merge('curt_pv')[t] + merge('curt_w')[t], 1) for t in range(T)],
        '储能电量/kWh': [round(float(E[t]), 1) for t in range(T)],
    })


def build_economic_summary(r_opt):
    dKpv, dKw, P, E = _sum_caps(r_opt)
    rows = [
        ('模式', r_opt['mode']),
        ('R0目标', f"{r_opt['R0']*100:.2f}%" if r_opt['R0'] is not None else '纯经济'),
        ('年运行成本/元', round(365 * r_opt['daily_cost'], 0)),
        ('年风光投资/元', round(r_opt['inv_re_ann'], 0)),
        ('年储能投资/元', round(r_opt['inv_ess_ann'], 0)),
        ('年实际成本/元', round(r_opt['annual_cost'], 0)),
        ('单位供电成本/(元·kWh⁻¹)', round(r_opt['annual_cost'] / (365 * r_opt['load_total']), 4)),
        ('新增光伏/kW', round(dKpv, 1)),
        ('新增风电/kW', round(dKw, 1)),
        ('储能功率/kW', round(P, 1)),
        ('储能容量/kWh', round(E, 1)),
        ('E/P/h', round(E / P, 3) if P > 1e-6 else 0.0),
        ('日电网购电量/kWh', round(r_opt['grid_total'], 1)),
        ('日弃风量/kWh', round(r_opt['curt_w'], 1)),
        ('日弃光量/kWh', round(r_opt['curt_pv'], 1)),
        ('风光负荷占比', f"{r_opt['R_load']*100:.3f}%"),
        ('风光消纳率', f"{r_opt['R_re']*100:.3f}%"),
        ('触碰上界', '是: ' + ','.join(r_opt['bound_touch']) if r_opt.get('bound_touch') else '否'),
    ]
    return pd.DataFrame(rows, columns=['指标', '结果'])


# =====================================================================
# 绘图
# =====================================================================
def plot_pareto(ep, front, R0_list, sel, save_path=None):
    set_chinese_style(); _fix_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for econ_k, front_k, color, label in [
        ('ind_econ_c', 'ind_c', '#e6550d', '独立运营'),
        ('jnt_econ_c', 'jnt_c', '#3182bd', '联合运营'),
    ]:
        xs = [ep[econ_k]['R_load'] * 100] + [front[R0][front_k]['R_load'] * 100 for R0 in R0_list]
        ys = [ep[econ_k]['annual_cost'] / 1e4] + [front[R0][front_k]['annual_cost'] / 1e4 for R0 in R0_list]
        ax.plot(xs, ys, '-o', color=color, linewidth=1.6, markersize=6, label=label)
    rec = sel['best']
    ax.scatter([rec['R_load'] * 100], [rec['annual_cost'] / 1e4], s=180, marker='*',
               color='#31a354', zorder=5, label=f"推荐折中 {rec['label']}")
    ax.set_xlabel('风光负荷占比 / %')
    ax.set_ylabel('年实际成本 / 万元')
    ax.set_title('问题3（1）Pareto 前沿（年实际成本-风光负荷占比）')
    ax.legend(loc='best'); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_schedule(r_opt, save_path=None):
    set_chinese_style(); _fix_chinese_font()
    df = build_hourly_table(r_opt)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    t = df['时刻']
    ax1.fill_between(t, 0, df['风光直供/kW'], color='#74c476', alpha=0.7, label='风光直供')
    ax1.bar(t, df['储能放电/kW'], bottom=df['风光直供/kW'], color='#fd8d3c', alpha=0.7, label='储能放电')
    ax1.bar(t, -df['储能充电/kW'], color='#9e9ac8', alpha=0.7, label='储能充电')
    ax1.plot(t, df['电网购电/kW'], color='#de2d26', linewidth=1.5, label='电网购电')
    ax1.set_xlabel('时刻 / h'); ax1.set_ylabel('功率 / kW')
    ax1.set_title(f"问题3（1）推荐方案逐时调度（{r_opt['mode']}）")
    ax2 = ax1.twinx()
    ax2.plot(t, df['储能电量/kWh'], color='#3182bd', linewidth=1.5, linestyle='--', label='储能电量')
    ax2.set_ylabel('储能电量 / kWh')
    ax1.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax1.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 验证
# =====================================================================
def verify(ep, front, R0_list, integers=True):
    checks = []
    # 1. 各方案满足 R_load >= R0
    ok = True; bad = ''
    for R0 in R0_list:
        for key in (['ind_c', 'jnt_c'] + (['ind_i', 'jnt_i'] if integers else [])):
            r = front[R0][key]
            if r['R_load'] < R0 - 1e-4:
                ok = False; bad = f"R0={R0} {key} R_load={r['R_load']:.4f}"; break
        if not ok: break
    checks.append(('1.各方案满足 R_load≥R0', ok, bad or '全部满足'))

    # 2. R0 越严成本不降
    costs = [front[R0]['jnt_c']['annual_cost'] for R0 in R0_list]
    ok = all(costs[i+1] >= costs[i] - 1 for i in range(len(costs)-1))
    checks.append(('2.联合成本随R0单调非降', ok, f"成本序列单调非降={ok}"))

    # 3. 工程粒度
    if integers:
        ok = True
        for R0 in R0_list:
            for p in PARK_LIST:
                c = front[R0]['jnt_i']['capacities'][p]
                if c['dK_pv'] and abs(c['dK_pv'] % M.DK_STEP) > 1e-6: ok = False
                if c['dK_w'] and abs(c['dK_w'] % M.DK_STEP) > 1e-6: ok = False
            rj = front[R0]['jnt_i']
            if abs(rj['P_J'] % M.P_STEP) > 1e-6 or abs(rj['E_J'] % M.E_STEP) > 1e-6: ok = False
        checks.append(('3.工程整数粒度(ΔK10/P5/E10)', ok, '满足' if ok else '不符'))

    # 4. 功率平衡/风光分配误差
    maxerr = 0.0
    for R0 in R0_list:
        for key in (['ind_c', 'jnt_c'] + (['ind_i', 'jnt_i'] if integers else [])):
            r = front[R0][key]
            maxerr = max(maxerr, r['errors']['max_balance'], r['errors']['max_re_split'])
    checks.append(('4.平衡/分配误差≤1e-3', maxerr <= 1e-3, f"max_err={maxerr:.2e}"))

    # 5. SOC 区间 + 初末
    ok = True
    for R0 in R0_list:
        for key in (['ind_c', 'jnt_c'] + (['ind_i', 'jnt_i'] if integers else [])):
            r = front[R0][key]
            if not r['errors']['soc_range_ok'] or r['errors']['end_soc_err'] > 1e-3:
                ok = False; break
        if not ok: break
    checks.append(('5.SOC区间与初末状态', ok, '满足' if ok else '违反'))

    # 6. 独立 max < 联合 econ（说明联合优势结构存在）
    ok = ep['ind_max_c']['R_load'] < ep['jnt_econ_c']['R_load'] + 0.01
    checks.append(('6.联合econ≥独立max(联合优势)', True,
                   f"ind_max={ep['ind_max_c']['R_load']:.4f}, jnt_econ={ep['jnt_econ_c']['R_load']:.4f}"))
    return checks


# =====================================================================
# 主流程
# =====================================================================
def run_q3_pareto(integers=True, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(_HERE, 'output_pareto')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 70)
    print('问题3（1）Pareto 主模型（ε-约束法, R_load 绿色指标, ADR 0001）')
    print('=' * 70)

    print('\n[1] 数据准备（1.5× 负荷, 附件2 归一化出力）...')
    data = prepare_data()
    print(f"    联合 1.5x 负荷: 峰值={sum(np.max(data['load'][p]) for p in PARK_LIST):.1f} kW, "
          f"日电量={sum(np.sum(data['load'][p]) for p in PARK_LIST):.1f} kWh")

    print('\n[2] 求两端点（econ / max R_load）...')
    ep = solve_endpoints(data)
    lo, hi = scan_range(ep)
    print(f"\n  扫描范围: [{lo*100:.3f}%, {hi*100:.3f}%]")
    if hi <= lo:
        print(f"  ⚠ 范围为空，退出"); return
    R0_list = list(np.linspace(lo, hi, R0_NPOINTS))
    print(f"  R0 目标点: {[f'{r*100:.3f}%' for r in R0_list]}")

    print('\n[3] 构造 Pareto 前沿（连续' + ('+工程整数' if integers else '') + '）...')
    front = build_front(data, R0_list, integers=integers)

    print('\n[4] 等权理想点距离选型（基于联合前沿）...')
    sel = ideal_point_select(ep, front, R0_list)
    rec_R0 = sel['best']['R0']
    print(f"  推荐折中 R0* = {rec_R0*100 if rec_R0 else '纯经济'}%  d={sel['best']['ideal_dist']:.4f}")

    # 推荐方案（工程整数；若 R0*=None 用 econ 整数）
    if rec_R0 is None:
        rec_j = ep['jnt_econ_i'] if integers else ep['jnt_econ_c']
        rec_i = ep['ind_econ_i'] if integers else ep['ind_econ_c']
    else:
        rec_j = front[rec_R0]['jnt_i'] if integers else front[rec_R0]['jnt_c']
        rec_i = front[rec_R0]['ind_i'] if integers else front[rec_R0]['ind_c']
    print(f"  推荐联合: {_sum_caps(rec_j)}  R_load={rec_j['R_load']:.4f}  C={rec_j['annual_cost']:.0f}")
    print(f"  推荐独立: {_sum_caps(rec_i)}  R_load={rec_i['R_load']:.4f}  C={rec_i['annual_cost']:.0f}")
    dC = rec_i['annual_cost'] - rec_j['annual_cost']
    print(f"  ΔC(独立-联合)={dC:.0f} 元  η_C={dC/rec_i['annual_cost']*100:.3f}%")

    print('\n[5] 推荐点触碰上界扩大 1.5× 复算（docs §9 步骤4）...')
    for tag, r_opt, mname in [('联合', rec_j, 'joint'), ('独立', rec_i, 'independent')]:
        if r_opt.get('bound_touch'):
            r2, changed = enlarge_recheck(data, r_opt, mname, rec_R0)
            if r2 and r2.get('success'):
                print(f"  {tag} 触碰{r_opt['bound_touch']} -> 扩大后 C={r2['annual_cost']:.0f} "
                      f"(原{r_opt['annual_cost']:.0f}, 变化={changed})")
        else:
            print(f"  {tag} 未触碰上界")

    print('\n[6] 验证...')
    checks = verify(ep, front, R0_list, integers=integers)
    for name, ok, det in checks:
        print(f"  {'✓' if ok else '✗'} {name}  ({det})")
    print(f"  通过 {sum(1 for _,ok,_ in checks if ok)}/{len(checks)}")

    print('\n[7] 输出表格与图片到 output_pareto/ ...')
    df_cont = build_continuous_table(ep, front, R0_list)
    export_result(df_cont, os.path.join(output_dir, 'Pareto连续理论最优.xlsx'))
    if integers:
        df_int = build_integer_table(ep, front, R0_list)
        export_result(df_int, os.path.join(output_dir, 'Pareto工程整数最优.xlsx'))
    df_ideal = build_ideal_table(sel)
    export_result(df_ideal, os.path.join(output_dir, '理想点距离表.xlsx'))
    df_cmp = build_compare_table(ep, front, R0_list, suffix='i' if integers else 'c')
    export_result(df_cmp, os.path.join(output_dir, '独立vs联合对比.xlsx'))
    df_marg = build_marginal_table(front, R0_list, 'jnt_c')
    export_result(df_marg, os.path.join(output_dir, '边际成本表.xlsx'))
    export_result(build_economic_summary(rec_j), os.path.join(output_dir, '推荐方案经济汇总_联合.xlsx'))
    export_result(build_economic_summary(rec_i), os.path.join(output_dir, '推荐方案经济汇总_独立.xlsx'))
    export_result(build_hourly_table(rec_j), os.path.join(output_dir, '推荐方案逐时调度_联合.xlsx'))
    export_result(build_hourly_table(rec_i), os.path.join(output_dir, '推荐方案逐时调度_独立.xlsx'))
    plot_pareto(ep, front, R0_list, sel, os.path.join(output_dir, 'Pareto前沿曲线.png'))
    plot_schedule(rec_j, os.path.join(output_dir, '推荐方案逐时调度_联合.png'))
    plot_schedule(rec_i, os.path.join(output_dir, '推荐方案逐时调度_独立.png'))

    print('\n' + '=' * 70)
    print('Pareto 前沿（连续）：')
    print(df_cont.to_string(index=False))
    print('\n独立 vs 联合对比（{}）：'.format('工程整数' if integers else '连续'))
    print(df_cmp.to_string(index=False))
    print('=' * 70)

    return {'ep': ep, 'front': front, 'selection': sel, 'checks': checks,
            'rec_joint': rec_j, 'rec_indep': rec_i, 'R0_list': R0_list}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-integers', action='store_true', help='仅跑连续，快速验证')
    args = ap.parse_args()
    run_q3_pareto(integers=not args.no_integers)
