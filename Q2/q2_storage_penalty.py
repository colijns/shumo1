# -*- coding: utf-8 -*-
"""
问题2（2）带弃电惩罚的平行结果集（View C，ADR 0003）

与无惩罚版 q2_storage.py 完全隔离：代码独立、输出到 Q2/output_penalty/。
共享 MILP (Q1/milp_model.py) 通过新增的向后兼容参数 curt_penalty=0.0 接入惩罚，
默认 0 时 Q1/Q2(1)/Q2(2) 无惩罚版行为不变（已由回归测试守护）。

目标函数（View C）：
    C_day(λ) = Σ_t [0.4(P_pv,L+P_pv,ch) + 0.5(P_w,L+P_w,ch) + P_grid
                + λ(P_curt,pv + P_curt,w)] Δt
    C_ann(λ) = 365·C_day(λ) + C_inv,ann
λ ∈ {0, 0.3, 0.6} 元/kWh。λ=0 退化为 View A，须复现 0/0=5514296.083。

口径（ADR 0003）：
  - 基准（联合无储能）也按 λ 计弃电惩罚，所有方案同口径比较；
  - 每个 λ 各跑完整 13468 点 2D 网格出热力图；
  - 保留充电成本 c 敏感性 + 投资价 s 敏感性（每个 λ 各一套）；
  - 新增 λ 敏感性主线：最优容量/成本/弃电 vs λ。

运行：
    $env:PYTHONUTF8='1'; & <math-python> Q2/q2_storage_penalty.py
    & <math-python> -m pytest Q2/test_q2_2_penalty_verify.py -v
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ----- 跨目录复用 Q1 / Q2 / templates -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_Q2_DIR = _HERE
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'templates'))
for _d in (_Q1_DIR, _Q2_DIR, _TEMPLATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import (  # noqa: E402
    load_load_data, load_solar_wind_data,
    C_PV, C_W, C_G, C_P_ESS, C_E_ESS, Y, ETA_C, ETA_D, DT,
)
from milp_model import (  # noqa: E402
    solve_park_optimization,
    run_no_storage as _run_no_storage,
    run_fixed_storage as _run_fixed_storage,
    run_engineering_storage,
)
from grid_search import grid_search_capacity  # noqa: E402
from q2_main import build_joint_profile, _fix_chinese_font  # noqa: E402
from common.io_utils import export_result  # noqa: E402
from common.plot_style import set_chinese_style, save_fig  # noqa: E402

T = 24
# 工程粒度与物理充分上界：
# 最大逐时风光余电为 448.115 kW，按 5 kW 向上取整得到 450 kW；
# 全日风光余电 1237.175 kWh，经 ηc=0.95、可用 SOC 区间 0.8 折算为
# 1469.145 kWh，按 10 kWh 向上取整得到 1470 kWh。
P_STEP, E_STEP = 5, 10
P_MAX, E_MAX = 450, 1470
GRID_POINT_COUNT = (P_MAX // P_STEP + 1) * (E_MAX // E_STEP + 1)
SURPLUS_ONLY_CHARGE = True
TOL = 1e-4

# 弃电惩罚因子 λ（元/kWh）；λ=0 为 View A 对照
LAMBDA_VALUES = [0.0, 0.3, 0.6]

# Q2(1) 联合无储能基准（View A，λ=0 验收口径）
JOINT_NO_STORAGE_DAILY = 15107.661
JOINT_NO_STORAGE_ANNUAL = 5514296.083
JOINT_NO_STORAGE_DAILY_CURT = 1237.175  # 无储能日弃风弃光 kWh（固定）


# =====================================================================
# 求解（每个 λ 各一套）
# =====================================================================

def _attach_capacity_diagnostics(result):
    """标记最优解是否触碰容量搜索上界，防止再次把截断解当成最终解。"""
    result['capacity_bounds'] = (P_MAX, E_MAX)
    if result.get('success'):
        result['hits_p_upper'] = abs(result['P_ess'] - P_MAX) <= TOL
        result['hits_e_upper'] = abs(result['E_ess'] - E_MAX) <= TOL
        result['hits_capacity_upper'] = (
            result['hits_p_upper'] or result['hits_e_upper']
        )
    else:
        result['hits_p_upper'] = False
        result['hits_e_upper'] = False
        result['hits_capacity_upper'] = False
    return result


def run_no_storage(load, G_pv, G_w, curt_penalty=0.0):
    """问题2（2）无储能基准；显式启用余电充电口径（P=E=0 时不改变结果）。"""
    return _run_no_storage(
        load, G_pv, G_w,
        curt_penalty=curt_penalty,
        surplus_only_charge=SURPLUS_ONLY_CHARGE,
    )


def run_fixed_storage(load, G_pv, G_w, P_ess=50, E_ess=100,
                      curt_penalty=0.0):
    """问题2（2）固定储能方案；储能只能吸收当小时风光余电。"""
    return _run_fixed_storage(
        load, G_pv, G_w,
        P_ess=P_ess, E_ess=E_ess,
        curt_penalty=curt_penalty,
        surplus_only_charge=SURPLUS_ONLY_CHARGE,
    )


def solve_continuous(L_J, G_pv_J, G_w_J, lam=0.0,
                     charge_cost=None, c_p_ess=None, c_e_ess=None):
    """连续 MILP（含弃电惩罚 λ）：P_ess/E_ess 连续变量。"""
    result = solve_park_optimization(
        L_J, G_pv_J, G_w_J,
        optimize_capacity=True, capacity_bounds=(P_MAX, E_MAX),
        charge_cost=charge_cost, c_p_ess=c_p_ess, c_e_ess=c_e_ess,
        curt_penalty=lam,
        surplus_only_charge=SURPLUS_ONLY_CHARGE,
    )
    return _attach_capacity_diagnostics(result)


def solve_integer(L_J, G_pv_J, G_w_J, lam=0.0):
    """工程整数 MILP（含弃电惩罚 λ）：P=5·n_P, E=10·n_E。"""
    result = run_engineering_storage(
        L_J, G_pv_J, G_w_J,
        P_step=P_STEP, E_step=E_STEP, P_max=P_MAX, E_max=E_MAX,
        curt_penalty=lam,
        surplus_only_charge=SURPLUS_ONLY_CHARGE,
    )
    return _attach_capacity_diagnostics(result)


def grid_search_2d(L_J, G_pv_J, G_w_J, lam=0.0, verbose=True):
    """2D 网格枚举（含弃电惩罚 λ）：91×148=13468 点。

    返回 dict 同 q2_storage.grid_search_2d，但 daily_cost / annual_cost 含惩罚。
    """
    gs = grid_search_capacity(
        L_J, G_pv_J, G_w_J,
        P_range=(0, P_MAX, P_STEP), E_range=(0, E_MAX, E_STEP),
        verbose=verbose, curt_penalty=lam,
        surplus_only_charge=SURPLUS_ONLY_CHARGE,
    )
    grid = gs['grid']

    pos = [p for p in grid if p['P_ess'] > 0 and p['E_ess'] > 0]
    gs['best_positive'] = min(pos, key=lambda x: x['annual_cost']) if pos else None

    P_vals = np.arange(0, P_MAX + P_STEP, P_STEP)
    E_vals = np.arange(0, E_MAX + E_STEP, E_STEP)
    cost_mat = np.full((len(P_vals), len(E_vals)), np.nan)
    lookup = {(p['P_ess'], p['E_ess']): p['annual_cost'] for p in grid}
    for i, p in enumerate(P_vals):
        for j, e in enumerate(E_vals):
            v = lookup.get((p, e))
            if v is not None:
                cost_mat[i, j] = v
    gs['cost_matrix'] = cost_mat
    gs['P_vals'] = P_vals
    gs['E_vals'] = E_vals
    gs['lam'] = lam
    return gs


def sensitivity_charge_cost(L_J, G_pv_J, G_w_J, lam=0.0, c_values=None):
    """充电成本敏感性（在固定 λ 下扫 c）：每个 c 求连续最优，记录 (P, E)。

    展示惩罚下盈亏平衡充电价如何随 λ 移动。
    """
    if c_values is None:
        c_values = [0.0, 0.1, 0.2, 0.3, 0.32, 0.35, 0.4, 0.5]
    rows = []
    for c in c_values:
        r = solve_continuous(L_J, G_pv_J, G_w_J, lam=lam, charge_cost=(c, c))
        if r['success']:
            rows.append({
                'c': c,
                'P_ess': r['P_ess'], 'E_ess': r['E_ess'],
                'daily_cost': r['daily_cost'],
                'daily_cost_pure': r['daily_cost_pure'],
                'daily_curt_penalty': r['daily_curt_penalty'],
                'annual_cost': r['annual_cost'],
                'curt_total': r['curt_total'],
            })
        else:
            rows.append({'c': c, 'P_ess': None, 'E_ess': None,
                         'daily_cost': None, 'daily_cost_pure': None,
                         'daily_curt_penalty': None, 'annual_cost': None,
                         'curt_total': None})
    return rows


def sensitivity_investment(grid_points, lam=0.0, s_values=None):
    """投资价格敏感性（在固定 λ 下扫 s）：复用该 λ 网格的 daily_cost（含惩罚）。

    daily_cost(λ) 在 (P,E) 处与投资价无关，只需重算年综合成本。
    """
    if s_values is None:
        s_values = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    rows = []
    for s in s_values:
        cp, ce = s * C_P_ESS, s * C_E_ESS
        best = None
        for p in grid_points:
            ann = 365 * p['daily_cost'] + (cp * p['P_ess'] + ce * p['E_ess']) / Y
            if best is None or ann < best['annual_cost']:
                best = {'P_ess': p['P_ess'], 'E_ess': p['E_ess'],
                        'annual_cost': ann, 'daily_cost': p['daily_cost']}
        rows.append({'s': s, 'c_p_ess': cp, 'c_e_ess': ce, 'lam': lam, **best})
    return rows


def sensitivity_lambda(L_J, G_pv_J, G_w_J, lam_values=None,
                       r_cont_by_lam=None):
    """λ 敏感性主线：每个 λ 取连续最优，记录最优容量/成本/弃电/消纳率。

    r_cont_by_lam: 可选 {lam: r_cont} 复用已解结果，避免重复求解。
    """
    if lam_values is None:
        lam_values = LAMBDA_VALUES
    rows = []
    for lam in lam_values:
        if r_cont_by_lam is not None and lam in r_cont_by_lam:
            r = r_cont_by_lam[lam]
        else:
            r = solve_continuous(L_J, G_pv_J, G_w_J, lam=lam)
        # 基准（无储能）同 λ 下的年综合，用于计算储能节省
        r_no = run_no_storage(L_J, G_pv_J, G_w_J, curt_penalty=lam)
        rows.append({
            'lam': lam,
            'P_ess': r['P_ess'], 'E_ess': r['E_ess'],
            'daily_cost': r['daily_cost'],
            'daily_cost_pure': r['daily_cost_pure'],
            'daily_curt_penalty': r['daily_curt_penalty'],
            'inv_annual': r['inv_annual'],
            'annual_cost': r['annual_cost'],
            'curt_total': r['curt_total'],
            're_ratio': r['re_ratio'],
            'baseline_annual': r_no['annual_cost'],
            'baseline_curt': r_no['curt_total'],
            'annual_saving': r_no['annual_cost'] - r['annual_cost'],
        })
    return rows


# =====================================================================
# 表格
# =====================================================================

def _row_from_result(scheme, r, lam):
    """方案行：含弃电惩罚分解。λ=0 时 日弃电惩罚=0、纯=含，与无惩罚表一致。"""
    return {
        'λ': lam,
        '方案': scheme,
        '储能功率/kW': r['P_ess'],
        '储能容量/kWh': r['E_ess'],
        '日运行成本(不含惩)/元': r['daily_cost_pure'],
        '日弃电惩罚/元': r['daily_curt_penalty'],
        '日运行成本/元': r['daily_cost'],
        '年均投资/元': r['inv_annual'],
        '年综合成本/元': r['annual_cost'],
        '日购电量/kWh': r['grid_total'],
        '日弃风弃光/kWh': r['curt_total'],
        '风光消纳率': r['re_ratio'],
    }


def build_scheme_table(r_no, r_fixed, r_cont, r_int, best_positive,
                       L_J, G_pv_J, G_w_J, lam):
    """单 λ 方案对比表。"""
    rows = [
        _row_from_result('联合无储能', r_no, lam),
        _row_from_result('固定50kW/100kWh', r_fixed, lam),
        _row_from_result('连续优化', r_cont, lam),
        _row_from_result('工程整数优化', r_int, lam),
    ]
    if best_positive is not None:
        r_bp = run_fixed_storage(L_J, G_pv_J, G_w_J,
                                 P_ess=best_positive['P_ess'],
                                 E_ess=best_positive['E_ess'],
                                 curt_penalty=lam)
        rows.append(_row_from_result('最优正配置(P>0,E>0)', r_bp, lam))
    return pd.DataFrame(rows)


def build_hourly_table(r_opt, L_J, G_pv_J, G_w_J):
    """24h 逐时运行表（最优配置下）。"""
    soc_full = list(r_opt['E'])  # 25 点 t=0..24
    df = pd.DataFrame({
        '时刻': list(range(1, T + 1)),
        '联合负荷/kW': L_J,
        '联合光伏/kW': G_pv_J,
        '联合风电/kW': G_w_J,
        '光伏直供/kW': r_opt['P_load_pv'],
        '风电直供/kW': r_opt['P_load_w'],
        '储能充电/kW': r_opt['P_ch'],
        '储能放电/kW': r_opt['P_dis'],
        '电网购电/kW': r_opt['P_grid'],
        '弃光/kW': r_opt['P_curt_pv'],
        '弃风/kW': r_opt['P_curt_w'],
        '储能电量(时段末)/kWh': soc_full[1:],
    })
    return df, soc_full


def build_lambda_table(sens_lam):
    """λ 敏感性总表。"""
    rows = []
    for s in sens_lam:
        rows.append({
            'λ': s['lam'],
            '最优功率/kW': s['P_ess'],
            '最优容量/kWh': s['E_ess'],
            '日运行成本(不含惩)/元': s['daily_cost_pure'],
            '日弃电惩罚/元': s['daily_curt_penalty'],
            '日运行成本/元': s['daily_cost'],
            '年均投资/元': s['inv_annual'],
            '年综合成本/元': s['annual_cost'],
            '日弃风弃光/kWh': s['curt_total'],
            '风光消纳率': s['re_ratio'],
            '基准年综合(无储能)/元': s['baseline_annual'],
            '年综合节省/元': s['annual_saving'],
        })
    return pd.DataFrame(rows)


# =====================================================================
# 绘图
# =====================================================================

def _lam_tag(lam):
    """文件名用 λ 标签。"""
    return f'lambda_{lam:.1f}'


def plot_heatmap(gs, lam, save_path=None):
    """2D 年综合成本热力图（含惩罚）：颜色相对该 λ 全局最优的增量。"""
    set_chinese_style()
    _fix_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 6))
    P_vals, E_vals = gs['P_vals'], gs['E_vals']
    cost = gs['cost_matrix']
    base = np.nanmin(cost)  # 相对全局最优着色（λ=0 时即 0/0）
    im = ax.pcolormesh(E_vals, P_vals, cost - base,
                       cmap='viridis', shading='auto')
    best = gs['best']
    ax.scatter([best['E_ess']], [best['P_ess']], marker='*', s=220, c='red',
               edgecolors='white', linewidths=1.2, zorder=5,
               label=f"全局最优 {best['P_ess']:.0f}/{best['E_ess']:.0f}")
    bp = gs.get('best_positive')
    if bp is not None and not (bp['P_ess'] == best['P_ess']
                               and bp['E_ess'] == best['E_ess']):
        ax.scatter([bp['E_ess']], [bp['P_ess']], marker='D', s=90, c='orange',
                   edgecolors='white', linewidths=1.0, zorder=5,
                   label=f"最优正配置 {bp['P_ess']:.0f}/{bp['E_ess']:.0f}")
    ax.set_xlabel('储能容量 E (kWh)')
    ax.set_ylabel('储能功率 P (kW)')
    ax.set_title(f'Q2(2) 带弃电惩罚 年综合成本热力图 (λ={lam:.1f} 元/kWh)\n'
                 f'(颜色=相对全局最优的年成本增量，元)')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('年综合成本增量 (元)')
    ax.legend(loc='upper right', fontsize=9)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_capacity_cost_curve(gs, lam, save_path=None):
    """容量-年综合成本曲线（含惩罚）：每个 E 取最优 P。"""
    set_chinese_style()
    _fix_chinese_font()
    grid = gs['grid']
    by_E = {}
    for p in grid:
        e = p['E_ess']
        if e not in by_E or p['annual_cost'] < by_E[e]:
            by_E[e] = p['annual_cost']
    Es = np.array(sorted(by_E.keys()))
    Cs = np.array([by_E[e] for e in Es])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Es, Cs, '-', lw=1.6, color='#2c7bb6')
    best = gs['best']
    ax.scatter([best['E_ess']], [best['annual_cost']], marker='*', s=200,
               c='red', zorder=5, label=f"全局最优 {best['P_ess']:.0f}/{best['E_ess']:.0f}")
    ax.set_xlabel('储能容量 E (kWh)')
    ax.set_ylabel('年综合成本 (元)')
    ax.set_title(f'Q2(2) 容量-年综合成本曲线 (λ={lam:.1f} 元/kWh)\n'
                 f'(每个容量取最优功率 P 下的年综合成本)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_sensitivity_charge(rows, lam, save_path=None):
    """充电成本敏感性（固定 λ）：最优容量 vs c。"""
    set_chinese_style()
    _fix_chinese_font()
    cs = [r['c'] for r in rows]
    Ps = [r['P_ess'] or 0 for r in rows]
    Es = [r['E_ess'] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, Es, '-o', lw=1.6, color='#d7191c', label='最优容量 E')
    ax.plot(cs, Ps, '--s', lw=1.4, color='#2c7bb6', label='最优功率 P')
    ax.set_xlabel('储能充电能量价 c (元/kWh)')
    ax.set_ylabel('最优配置 (kW / kWh)')
    ax.set_title(f'Q2(2) 充电成本敏感性 (λ={lam:.1f} 元/kWh)\n'
                 f'(固定惩罚下，最优容量随充电价的变化)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_sensitivity_investment(rows, lam, save_path=None):
    """投资价敏感性（固定 λ）：最优容量 vs s。"""
    set_chinese_style()
    _fix_chinese_font()
    ss = [r['s'] for r in rows]
    Ps = [r['P_ess'] for r in rows]
    Es = [r['E_ess'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ss, Es, '-o', lw=1.6, color='#d7191c', label='最优容量 E')
    ax.plot(ss, Ps, '--s', lw=1.4, color='#2c7bb6', label='最优功率 P')
    ax.axvline(1.0, ls=':', color='gray')
    ax.text(1.01, max(Es) * 0.5 if max(Es) > 0 else 50,
            '基准 s=1', fontsize=9, color='gray')
    ax.set_xlabel('储能投资价格缩放因子 s (s=1 为基准)')
    ax.set_ylabel('最优配置 (kW / kWh)')
    ax.set_title(f'Q2(2) 投资价格敏感性 (λ={lam:.1f} 元/kWh)\n'
                 f'(固定惩罚下，最优容量随投资价的变化)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_lambda_sensitivity(sens_lam, save_path=None):
    """λ 敏感性主线：上=最优容量 vs λ；下=年综合成本(最优 vs 基准) vs λ。"""
    set_chinese_style()
    _fix_chinese_font()
    lams = [s['lam'] for s in sens_lam]
    Ps = [s['P_ess'] for s in sens_lam]
    Es = [s['E_ess'] for s in sens_lam]
    opt_ann = [s['annual_cost'] for s in sens_lam]
    base_ann = [s['baseline_annual'] for s in sens_lam]
    curts = [s['curt_total'] for s in sens_lam]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True)
    ax1.plot(lams, Es, '-o', lw=1.6, color='#d7191c', label='最优容量 E (kWh)')
    ax1.plot(lams, Ps, '--s', lw=1.4, color='#2c7bb6', label='最优功率 P (kW)')
    ax1.set_ylabel('最优配置')
    ax1.set_title('Q2(2) 弃电惩罚 λ 敏感性：最优容量与年综合成本随 λ\n'
                  '(λ=0 锚定 View A 的 0/0；λ↑ 推动最优配置转正)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(lams, base_ann, '-^', lw=1.6, color='gray', label='基准(无储能)年综合')
    ax2.plot(lams, opt_ann, '-o', lw=1.6, color='#1a9641', label='最优年综合')
    ax2.set_xlabel('弃电惩罚因子 λ (元/kWh)')
    ax2.set_ylabel('年综合成本 (元)')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    # 次轴：弃电量
    ax2r = ax2.twinx()
    ax2r.plot(lams, curts, ':D', lw=1.2, color='#d7191c', label='最优日弃电(kWh)')
    ax2r.set_ylabel('日弃风弃光 (kWh)', color='#d7191c')
    ax2r.tick_params(axis='y', labelcolor='#d7191c')
    ax2r.legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


# =====================================================================
# 验收（分层：λ=0 一致性 + λ>0 惩罚适用）
# =====================================================================

def verify_lambda_zero(r_cont, r_int, gs, sens_c, sens_inv, L_J, G_pv_J, G_w_J):
    """λ=0 层：复现 View A 关键检查（与无惩罚版 q2_storage.verify_q2_2 对齐）。"""
    checks = []
    e = r_cont['errors']

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    add('λ=0 功率平衡误差<1e-4',
        e['max_balance_error'] < TOL, f"max={e['max_balance_error']:.2e}")
    add('λ=0 风光分配误差<1e-4',
        e['max_re_error'] < TOL, f"max={e['max_re_error']:.2e}")
    # 不允许电网充电（结构性）
    ch_from_grid = 0.0
    for t in range(T):
        ch_from_grid = max(ch_from_grid,
                           abs(r_cont['P_ch'][t] - r_cont['P_ch_pv'][t] - r_cont['P_ch_w'][t]))
    add('λ=0 不允许电网充电(结构性)',
        ch_from_grid < TOL, f"充电=光伏+风电充电，偏差={ch_from_grid:.2e}")
    # 只用余电充电（显式约束：每小时充电不超过风光余电）
    surplus_ok = True
    for t in range(T):
        surplus = max(G_pv_J[t] + G_w_J[t] - L_J[t], 0.0)
        if r_cont['P_ch'][t] > surplus + TOL:
            surplus_ok = False
            break
    add('λ=0 只用余电充电(显式约束)', surplus_ok,
        '逐时充电功率均不超过风光余电' if surplus_ok else '存在超出风光余电的充电')
    # 不同时充放电
    add('λ=0 不同时充放电',
        all(not (r_cont['P_ch'][t] > TOL and r_cont['P_dis'][t] > TOL)
            for t in range(T)), 'z 互斥约束生效')
    # 年综合成本恒等
    recon = 365 * r_cont['daily_cost'] + r_cont['inv_annual']
    add('λ=0 年综合=365×日运行+年均投资',
        abs(recon - r_cont['annual_cost']) < 1e-2,
        f"重构={recon:.3f} vs {r_cont['annual_cost']:.3f}")
    # KEY: λ=0 复现 View A 的 0/0=5514296.083
    add('λ=0 复现 View A 年综合 5514296.083',
        abs(r_cont['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < 1e-2,
        f"年综合={r_cont['annual_cost']:.3f}")
    add('λ=0 最优为 0/0',
        r_cont['P_ess'] < 1e-3 and r_cont['E_ess'] < 1e-3,
        f"P={r_cont['P_ess']:.4f}, E={r_cont['E_ess']:.4f}")
    add('λ=0 弃电惩罚=0',
        abs(r_cont['daily_curt_penalty']) < TOL,
        f"daily_curt_penalty={r_cont['daily_curt_penalty']:.2e}")
    add('λ=0 daily_cost_pure=daily_cost',
        abs(r_cont['daily_cost_pure'] - r_cont['daily_cost']) < TOL,
        '惩罚为 0，二者相等')
    # 工程粒度
    add('λ=0 工程功率符合5kW粒度',
        abs(r_int['P_ess'] - round(r_int['P_ess'] / P_STEP) * P_STEP) < TOL,
        f"P={r_int['P_ess']:.1f}")
    add('λ=0 工程容量符合10kWh粒度',
        abs(r_int['E_ess'] - round(r_int['E_ess'] / E_STEP) * E_STEP) < TOL,
        f"E={r_int['E_ess']:.1f}")
    # 0/0 全局最优（网格）
    best = gs['best']
    add('λ=0 0/0 为全局最优(网格)',
        best['P_ess'] == 0 and best['E_ess'] == 0,
        f"网格最优={best['P_ess']:.0f}/{best['E_ess']:.0f}, "
        f"成本={best['annual_cost']:.2f}")
    # 充电价敏感性：c≥0.40 归零、c≤0.35 为正
    zero_above = all((r['E_ess'] or 0) < 1e-3 for r in sens_c if r['c'] >= 0.40)
    pos_below = all((r['E_ess'] or 0) > 1e-3 for r in sens_c if r['c'] <= 0.35)
    add('λ=0 盈亏平衡充电价∈(0.35,0.40]',
        zero_above and pos_below,
        'c≤0.35 为正、c≥0.40 归零' if (zero_above and pos_below) else '边界异常')
    # 投资价 s=1 归零
    base_row = next(r for r in sens_inv if r['s'] == 1.0)
    add('λ=0 投资价基准 s=1 容量为0',
        base_row['E_ess'] < 1e-3,
        f"s=1 最优={base_row['P_ess']:.0f}/{base_row['E_ess']:.0f}")
    return checks


def verify_lambda_positive(r_cont, r_int, r_no, gs, sens_c, lam,
                           L_J, G_pv_J, G_w_J):
    """λ>0 层：惩罚适用检查。"""
    checks = []
    e = r_cont['errors']

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    add(f'λ={lam} 功率平衡误差<1e-4',
        e['max_balance_error'] < TOL, f"max={e['max_balance_error']:.2e}")
    add(f'λ={lam} 风光分配误差<1e-4',
        e['max_re_error'] < TOL, f"max={e['max_re_error']:.2e}")
    surplus_ok = all(
        r_cont['P_ch'][t] <= max(G_pv_J[t] + G_w_J[t] - L_J[t], 0.0) + TOL
        for t in range(T)
    )
    add(f'λ={lam} 只用当小时风光余电充电',
        surplus_ok, '逐时显式约束生效')
    add(f'λ={lam} 不同时充放电',
        all(not (r_cont['P_ch'][t] > TOL and r_cont['P_dis'][t] > TOL)
            for t in range(T)), 'z 互斥约束生效')
    # 年综合恒等（含惩罚在 daily_cost）
    recon = 365 * r_cont['daily_cost'] + r_cont['inv_annual']
    add(f'λ={lam} 年综合=365×日运行(含惩)+年均投资',
        abs(recon - r_cont['annual_cost']) < 1e-2,
        f"重构={recon:.3f} vs {r_cont['annual_cost']:.3f}")
    # 惩罚分解恒等
    add(f'λ={lam} daily_cost=纯+惩罚',
        abs(r_cont['daily_cost'] - r_cont['daily_cost_pure']
            - r_cont['daily_curt_penalty']) < 1e-2,
        f"纯={r_cont['daily_cost_pure']:.3f}+惩={r_cont['daily_curt_penalty']:.3f}"
        f"={r_cont['daily_cost_pure']+r_cont['daily_curt_penalty']:.3f} vs {r_cont['daily_cost']:.3f}")
    # 基准年综合 = View A + 365·λ·1237.175
    expected_base = JOINT_NO_STORAGE_ANNUAL + 365 * lam * JOINT_NO_STORAGE_DAILY_CURT
    add(f'λ={lam} 基准年综合=5514296.083+365·λ·1237.175',
        abs(r_no['annual_cost'] - expected_base) < 1e-2,
        f"基准={r_no['annual_cost']:.3f} vs 预期={expected_base:.3f}")
    # 最优 ≤ 基准（储能总可不配 = 基准）
    add(f'λ={lam} 最优年综合≤基准',
        r_cont['annual_cost'] <= r_no['annual_cost'] + 1e-2,
        f"最优={r_cont['annual_cost']:.3f} ≤ 基准={r_no['annual_cost']:.3f}")
    # 最优弃电 ≤ 基准弃电
    add(f'λ={lam} 最优弃电≤基准弃电(1237.175)',
        r_cont['curt_total'] <= r_no['curt_total'] + 1e-2,
        f"最优弃电={r_cont['curt_total']:.3f} ≤ {r_no['curt_total']:.3f}")
    add(f'λ={lam} 连续最优未触碰物理容量上界',
        not r_cont['hits_capacity_upper'],
        f"P={r_cont['P_ess']:.3f}/{P_MAX}, E={r_cont['E_ess']:.3f}/{E_MAX}")
    # 消纳率 ≥ 基准
    add(f'λ={lam} 消纳率≥基准',
        r_cont['re_ratio'] >= r_no['re_ratio'] - 1e-6,
        f"最优={r_cont['re_ratio']:.4f} ≥ 基准={r_no['re_ratio']:.4f}")
    # 网格全局最优 = 连续最优（工程粒度容差）
    best = gs['best']
    add(f'λ={lam} 网格最优=连续最优(±工程粒度)',
        abs(best['P_ess'] - r_cont['P_ess']) <= P_STEP + TOL
        and abs(best['E_ess'] - r_cont['E_ess']) <= E_STEP + TOL,
        f"网格={best['P_ess']:.0f}/{best['E_ess']:.0f} vs 连续={r_cont['P_ess']:.3f}/{r_cont['E_ess']:.3f}")
    # 低 c（c=0）下容量为正：惩罚+免费充电共同促储能
    c0 = next(r for r in sens_c if r['c'] == 0.0)
    add(f'λ={lam} 低充电价(c=0)下容量为正',
        (c0['E_ess'] or 0) > 1e-3,
        f"c=0 最优={c0['P_ess']:.1f}/{c0['E_ess']:.1f}")
    # 基准弃电恒为 1237.175（无储能弃电与 λ 无关）
    add(f'λ={lam} 基准弃电=1237.175(与λ无关)',
        abs(r_no['curt_total'] - JOINT_NO_STORAGE_DAILY_CURT) < 1e-2,
        f"基准弃电={r_no['curt_total']:.3f}")
    return checks


# =====================================================================
# 主流程
# =====================================================================

def run_q2_2_penalty(output_dir=None):
    """问题2（2）带弃电惩罚完整求解流程（λ∈{0,0.3,0.6}）。"""
    if output_dir is None:
        output_dir = os.path.join(_HERE, 'output_penalty')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 64)
    print('问题2（2）带弃电惩罚（View C, ADR 0003）  λ∈{0, 0.3, 0.6}')
    print('=' * 64)

    print('\n[1] 读取数据 + 联合 profile...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)

    per_lambda = {}  # lam -> 结果字典
    r_cont_by_lam = {}

    for lam in LAMBDA_VALUES:
        tag = _lam_tag(lam)
        lam_dir = os.path.join(output_dir, tag)
        os.makedirs(lam_dir, exist_ok=True)
        print(f"\n{'='*64}\n[λ={lam}] 求解（输出 {tag}/）\n{'='*64}")

        print(f'  [a] 基准：联合无储能 + 固定50/100（含惩罚）...')
        r_no = run_no_storage(L_J, G_pv_J, G_w_J, curt_penalty=lam)
        r_fixed = run_fixed_storage(L_J, G_pv_J, G_w_J, P_ess=50, E_ess=100,
                                    curt_penalty=lam)
        print(f"      无储能: 年综合={r_no['annual_cost']:.3f}, "
              f"日弃电={r_no['curt_total']:.3f}, 日惩罚={r_no['daily_curt_penalty']:.3f}")
        print(f"      50/100: 年综合={r_fixed['annual_cost']:.3f}")

        print(f'  [b] 连续 MILP（含惩罚）...')
        t0 = time.time()
        r_cont = solve_continuous(L_J, G_pv_J, G_w_J, lam=lam)
        r_cont_by_lam[lam] = r_cont
        print(f"      最优: P={r_cont['P_ess']:.4f}, E={r_cont['E_ess']:.4f}, "
              f"年综合={r_cont['annual_cost']:.3f} ({time.time()-t0:.1f}s)")

        print(f'  [c] 工程整数 MILP (5kW/10kWh)...')
        t0 = time.time()
        r_int = solve_integer(L_J, G_pv_J, G_w_J, lam=lam)
        print(f"      最优: P={r_int['P_ess']:.1f}, E={r_int['E_ess']:.1f}, "
              f"年综合={r_int['annual_cost']:.3f} ({time.time()-t0:.1f}s)")

        print(f'  [d] 2D 网格枚举 ({GRID_POINT_COUNT} 点)...')
        t0 = time.time()
        gs = grid_search_2d(L_J, G_pv_J, G_w_J, lam=lam, verbose=True)
        bp = gs['best_positive']
        print(f"      网格最优: {gs['best']['P_ess']:.0f}/{gs['best']['E_ess']:.0f}, "
              f"成本={gs['best']['annual_cost']:.3f} ({time.time()-t0:.1f}s)")
        if bp:
            print(f"      最优正配置: {bp['P_ess']:.0f}/{bp['E_ess']:.0f}, "
                  f"成本={bp['annual_cost']:.3f}")

        print(f'  [e] 敏感性 (c 扫描 + s 扫描)...')
        sens_c = sensitivity_charge_cost(L_J, G_pv_J, G_w_J, lam=lam)
        sens_inv = sensitivity_investment(gs['grid'], lam=lam)
        print(f"      c 扫描: c=0→E={(sens_c[0]['E_ess'] or 0):.1f}, "
              f"c=0.4→E={(next(r for r in sens_c if r['c']==0.4)['E_ess'] or 0):.1f}")
        print(f"      s 扫描: s=1→E={next(r for r in sens_inv if r['s']==1.0)['E_ess']:.1f}")

        per_lambda[lam] = {
            'r_no': r_no, 'r_fixed': r_fixed, 'r_cont': r_cont, 'r_int': r_int,
            'grid_search': gs, 'sens_c': sens_c, 'sens_inv': sens_inv,
            'lam_dir': lam_dir,
        }

    print(f"\n{'='*64}\n[λ 敏感性主线]\n{'='*64}")
    sens_lam = sensitivity_lambda(L_J, G_pv_J, G_w_J, LAMBDA_VALUES,
                                  r_cont_by_lam=r_cont_by_lam)
    for s in sens_lam:
        print(f"  λ={s['lam']:.1f}: P={s['P_ess']:.2f}, E={s['E_ess']:.2f}, "
              f"年综合={s['annual_cost']:.3f}, 基准={s['baseline_annual']:.3f}, "
              f"节省={s['annual_saving']:.3f}, 弃电={s['curt_total']:.3f}, "
              f"消纳率={s['re_ratio']:.4f}")

    print(f"\n{'='*64}\n[验收]\n{'='*64}")
    all_checks = []
    # λ=0 层
    p0 = per_lambda[0.0]
    chk0 = verify_lambda_zero(p0['r_cont'], p0['r_int'], p0['grid_search'],
                              p0['sens_c'], p0['sens_inv'], L_J, G_pv_J, G_w_J)
    for name, ok, det in chk0:
        print(f"  {'✓' if ok else '✗'} {name}  ({det})")
    all_checks.extend(chk0)
    # λ>0 层
    for lam in LAMBDA_VALUES:
        if lam == 0.0:
            continue
        p = per_lambda[lam]
        chk = verify_lambda_positive(p['r_cont'], p['r_int'], p['r_no'],
                                     p['grid_search'], p['sens_c'], lam,
                                     L_J, G_pv_J, G_w_J)
        for name, ok, det in chk:
            print(f"  {'✓' if ok else '✗'} {name}  ({det})")
        all_checks.extend(chk)
    n_pass = sum(1 for _, p, _ in all_checks if p)
    print(f"  通过 {n_pass}/{len(all_checks)} 项检查")

    print(f"\n{'='*64}\n[输出表格与图片]\n{'='*64}")
    # 每个 λ 子文件夹
    scheme_all = []
    for lam in LAMBDA_VALUES:
        p = per_lambda[lam]
        d = p['lam_dir']
        df_scheme = build_scheme_table(p['r_no'], p['r_fixed'], p['r_cont'],
                                       p['r_int'], p['grid_search'].get('best_positive'),
                                       L_J, G_pv_J, G_w_J, lam)
        df_hourly, _ = build_hourly_table(p['r_cont'], L_J, G_pv_J, G_w_J)
        df_sens_c = pd.DataFrame(p['sens_c'])
        df_sens_inv = pd.DataFrame(p['sens_inv'])
        scheme_all.append(df_scheme)

        export_result(df_scheme, os.path.join(d, '方案对比表.xlsx'))
        export_result(df_hourly, os.path.join(d, '逐时运行表.xlsx'))
        export_result(df_sens_c, os.path.join(d, '充电成本敏感性.xlsx'))
        export_result(df_sens_inv, os.path.join(d, '投资价格敏感性.xlsx'))
        plot_heatmap(p['grid_search'], lam, os.path.join(d, '年综合成本热力图.png'))
        plot_capacity_cost_curve(p['grid_search'], lam, os.path.join(d, '容量成本曲线.png'))
        plot_sensitivity_charge(p['sens_c'], lam, os.path.join(d, '充电成本敏感性.png'))
        plot_sensitivity_investment(p['sens_inv'], lam, os.path.join(d, '投资价格敏感性.png'))
        print(f"  {d}/  (4 表 + 4 图)")

    # 顶层汇总
    df_scheme_all = pd.concat(scheme_all, ignore_index=True)
    df_lam = build_lambda_table(sens_lam)
    export_result(df_scheme_all, os.path.join(output_dir, '方案对比总表.xlsx'))
    export_result(df_lam, os.path.join(output_dir, 'λ敏感性总表.xlsx'))
    plot_lambda_sensitivity(sens_lam, os.path.join(output_dir, 'λ敏感性曲线.png'))
    print(f"  {output_dir}/  方案对比总表.xlsx + λ敏感性总表.xlsx + λ敏感性曲线.png")

    return {
        'per_lambda': per_lambda,
        'sens_lam': sens_lam,
        'checks': all_checks,
        'df_scheme_all': df_scheme_all,
        'df_lam': df_lam,
        'L_J': L_J, 'G_pv_J': G_pv_J, 'G_w_J': G_w_J,
    }


if __name__ == '__main__':
    run_q2_2_penalty()
    print('\nq2_storage_penalty 运行完毕')
