# -*- coding: utf-8 -*-
"""
问题2（2）：联合园区共享储能优化

核心思路（见 docs/问题2（2）.md + ADR 0001/0002）：
  - 复用 Q2(1) 的联合 profile（q2_main.build_joint_profile）；
  - 复用 Q1 的统一 MILP 容量优化分支（milp_model.solve_park_optimization）；
  - 充电成本口径 View A（ADR 0002）：储能充电按 0.4/0.5 计，与消纳同价；
  - "只用余电充电"由 View A 成本结构涌现，非显式约束（结构性禁电网充电已足够）；
  - 三步求解：连续 MILP -> 工程整数 MILP (5kW/10kWh) -> 2D 网格枚举；
  - 0/0 反直觉结论，用 2D 热力图 + 充电成本/投资价敏感性证明稳健。

运行：
    conda activate math
    python Q2/q2_storage.py
    python -m pytest Q2/test_q2_2_verify.py -v
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
    solve_park_optimization, run_no_storage, run_fixed_storage,
    run_engineering_storage,
)
from grid_search import grid_search_capacity  # noqa: E402
from q2_main import build_joint_profile, _fix_chinese_font  # noqa: E402
from common.io_utils import export_result  # noqa: E402
from common.plot_style import set_chinese_style, save_fig  # noqa: E402

T = 24
# 工程粒度与搜索上界（与 Q1 一致；5kW/10kWh 是 Q1 假设 21，非题目给定）
P_STEP, E_STEP = 5, 10
P_MAX, E_MAX = 200, 600
TOL = 1e-4

# Q2(1) 联合无储能基准（验收口径，来自 docs/问题2（1）.md §6）
JOINT_NO_STORAGE_DAILY = 15107.661
JOINT_NO_STORAGE_ANNUAL = 5514296.083


# =====================================================================
# 求解
# =====================================================================

def solve_continuous(L_J, G_pv_J, G_w_J,
                     charge_cost=None, c_p_ess=None, c_e_ess=None):
    """连续 MILP：P_ess/E_ess 为连续变量，搜索上界 (P_MAX, E_MAX)。"""
    return solve_park_optimization(
        L_J, G_pv_J, G_w_J,
        optimize_capacity=True, capacity_bounds=(P_MAX, E_MAX),
        charge_cost=charge_cost, c_p_ess=c_p_ess, c_e_ess=c_e_ess,
    )


def solve_integer(L_J, G_pv_J, G_w_J):
    """工程整数 MILP：P=5·n_P, E=10·n_E，上界 (P_MAX, E_MAX)。"""
    return run_engineering_storage(
        L_J, G_pv_J, G_w_J,
        P_step=P_STEP, E_step=E_STEP, P_max=P_MAX, E_max=E_MAX,
    )


def grid_search_2d(L_J, G_pv_J, G_w_J, verbose=True):
    """2D 网格枚举：P∈[0,200] step5, E∈[0,600] step10（41×61=2501 点）。

    返回 dict：
        'best'           : 全局最优 (含 (0,0))
        'best_positive'  : (P>0, E>0) 中年综合成本最低者
        'grid'           : 全部点 (P, E, daily_cost, annual_cost)
        'cost_matrix'    : annual_cost 二维矩阵 [P_idx, E_idx]
        'P_vals','E_vals': 坐标轴
    """
    gs = grid_search_capacity(
        L_J, G_pv_J, G_w_J,
        P_range=(0, P_MAX, P_STEP), E_range=(0, E_MAX, E_STEP),
        verbose=verbose,
    )
    grid = gs['grid']

    # 最优正配置：(P>0 且 E>0) 中年综合成本最低
    pos = [p for p in grid if p['P_ess'] > 0 and p['E_ess'] > 0]
    gs['best_positive'] = min(pos, key=lambda x: x['annual_cost']) if pos else None

    # 构造二维成本矩阵（便于热力图）
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
    return gs


def sensitivity_charge_cost(L_J, G_pv_J, G_w_J,
                            c_values=None):
    """充电成本敏感性：扫均匀充电价 c，每个求连续最优，记录最优 (P, E)。

    盈亏平衡充电价 c* ∈ (0.35, 0.40] 元/kWh：c≤0.35 时最优容量为正，
    c≥0.40 时为 0。题目光伏充电价 0.4 恰在边界、风电 0.5 更高，故 0/0 最优。
    直接回应 ADR 0002 的口径风险。
    """
    if c_values is None:
        c_values = [0.0, 0.1, 0.2, 0.3, 0.32, 0.35, 0.4, 0.5]
    rows = []
    for c in c_values:
        r = solve_continuous(L_J, G_pv_J, G_w_J, charge_cost=(c, c))
        if r['success']:
            rows.append({
                'c': c,
                'P_ess': r['P_ess'], 'E_ess': r['E_ess'],
                'daily_cost': r['daily_cost'], 'annual_cost': r['annual_cost'],
            })
        else:
            rows.append({'c': c, 'P_ess': None, 'E_ess': None,
                         'daily_cost': None, 'annual_cost': None})
    return rows


def sensitivity_investment(grid_points, s_values=None):
    """投资价格敏感性：扫投资缩放因子 s，复用 grid 的 daily_cost 重算年综合成本。

    无需重解 dispatch——daily_cost 只依赖 (P,E) 与充电价，与投资价无关。
    s=1 为基准（0/0）；s 降到 ~0.69 以下时储能变经济（C_E_ESS<1240 元/kWh）。
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
        rows.append({'s': s, 'c_p_ess': cp, 'c_e_ess': ce, **best})
    return rows


# =====================================================================
# 表格
# =====================================================================

def _row_from_result(scheme, r):
    return {
        '方案': scheme,
        '储能功率/kW': r['P_ess'],
        '储能容量/kWh': r['E_ess'],
        '日运行成本/元': r['daily_cost'],
        '年均投资/元': r['inv_annual'],
        '年综合成本/元': r['annual_cost'],
        '日购电量/kWh': r['grid_total'],
        '日弃风弃光/kWh': r['curt_total'],
        '风光消纳率': r['re_ratio'],
    }


def build_scheme_table(r_no, r_fixed, r_cont, r_int, best_positive, L_J, G_pv_J, G_w_J):
    """方案对比表：联合无储能 / 固定50/100 / 连续优化 / 工程整数 / 最优正配置。"""
    rows = [
        _row_from_result('联合无储能', r_no),
        _row_from_result('固定50kW/100kWh', r_fixed),
        _row_from_result('连续优化', r_cont),
        _row_from_result('工程整数优化', r_int),
    ]
    # 最优正配置：重解一次拿到完整指标
    if best_positive is not None:
        r_bp = run_fixed_storage(L_J, G_pv_J, G_w_J,
                                 P_ess=best_positive['P_ess'],
                                 E_ess=best_positive['E_ess'])
        row = _row_from_result('最优正配置(P>0,E>0)', r_bp)
        rows.append(row)
    return pd.DataFrame(rows)


def build_hourly_table(r_opt, L_J, G_pv_J, G_w_J):
    """24h 逐时运行表（最优配置下；0/0 时退化为联合无储能调度）。
    SOC 作为一列（时段末电量 E(t=1..24)）；另返回完整 25 点序列 E(t=0..24)。"""
    soc_full = list(r_opt['E'])  # 25 点 t=0..24，含初始
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
        '储能电量(时段末)/kWh': soc_full[1:],  # E(1..24)，与时刻行对齐
    })
    return df, soc_full


# =====================================================================
# 绘图
# =====================================================================

def plot_heatmap(gs, save_path=None):
    """2D 年综合成本热力图：P×E 网格，原点(0/0)为全局最优。"""
    set_chinese_style()
    _fix_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 6))
    P_vals, E_vals = gs['P_vals'], gs['E_vals']
    cost = gs['cost_matrix']
    # 用相对偏移着色（相对 0/0 的成本增量），凸显原点最优
    base = cost[0, 0]
    im = ax.pcolormesh(E_vals, P_vals, cost - base,
                       cmap='viridis', shading='auto')
    ax.scatter([0], [0], marker='*', s=220, c='red',
               edgecolors='white', linewidths=1.2, zorder=5, label='全局最优 0/0')
    bp = gs.get('best_positive')
    if bp is not None:
        ax.scatter([bp['E_ess']], [bp['P_ess']], marker='D', s=90, c='orange',
                   edgecolors='white', linewidths=1.0, zorder=5,
                   label=f"最优正配置 {bp['P_ess']:.0f}/{bp['E_ess']:.0f}")
    ax.set_xlabel('储能容量 E (kWh)')
    ax.set_ylabel('储能功率 P (kW)')
    ax.set_title('Q2(2) 联合共享储能 年综合成本热力图\n(颜色=相对 0/0 方案的年成本增量，元)')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('年综合成本增量 (元)')
    ax.legend(loc='upper right', fontsize=9)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_capacity_cost_curve(gs, save_path=None):
    """容量-年综合成本曲线：每个 E 取最优 P 下的年综合成本，展示最小值在 E=0。"""
    set_chinese_style()
    _fix_chinese_font()
    grid = gs['grid']
    # 每个 E 取最小的 annual_cost
    by_E = {}
    for p in grid:
        e = p['E_ess']
        if e not in by_E or p['annual_cost'] < by_E[e]:
            by_E[e] = p['annual_cost']
    Es = np.array(sorted(by_E.keys()))
    Cs = np.array([by_E[e] for e in Es])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Es, Cs, '-', lw=1.6, color='#2c7bb6')
    ax.scatter([0], [by_E[0]], marker='*', s=200, c='red', zorder=5,
               label='全局最优 0/0')
    ax.set_xlabel('储能容量 E (kWh)')
    ax.set_ylabel('年综合成本 (元)')
    ax.set_title('Q2(2) 容量-年综合成本曲线\n(每个容量取最优功率 P 下的年综合成本)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_sensitivity_charge(rows, save_path=None):
    """充电成本敏感性：最优容量 vs 充电价 c，标盈亏平衡点 ≈0.32。"""
    set_chinese_style()
    _fix_chinese_font()
    cs = [r['c'] for r in rows]
    Ps = [r['P_ess'] or 0 for r in rows]
    Es = [r['E_ess'] or 0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cs, Es, '-o', lw=1.6, color='#d7191c', label='最优容量 E')
    ax.plot(cs, Ps, '--s', lw=1.4, color='#2c7bb6', label='最优功率 P')
    ax.axvline(0.32, ls=':', color='gray')
    ax.text(0.325, max(Es) * 0.9 if max(Es) > 0 else 50,
            '盈亏平衡\n充电价≈0.32', fontsize=9, color='gray')
    ax.set_xlabel('储能充电能量价 c (元/kWh)')
    ax.set_ylabel('最优配置 (kW / kWh)')
    ax.set_title('Q2(2) 充电成本敏感性：最优容量随充电价跳变\n(c<0.32 容量为正；c≥0.32 为 0)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_sensitivity_investment(rows, save_path=None):
    """投资价敏感性：最优容量 vs 投资缩放因子 s。"""
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
            '基准 s=1\n(0/0 最优)', fontsize=9, color='gray')
    ax.set_xlabel('储能投资价格缩放因子 s (s=1 为基准)')
    ax.set_ylabel('最优配置 (kW / kWh)')
    ax.set_title('Q2(2) 投资价格敏感性：最优容量随投资价变化\n(s 降到约 0.69 以下，储能变经济)')
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


# =====================================================================
# 验收
# =====================================================================

def verify_q2_2(r_cont, r_int, gs, sens_c, sens_inv, L_J, G_pv_J, G_w_J):
    """覆盖 docs/问题2（2）.md §9 检查项。"""
    checks = []
    e = r_cont['errors']

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # 功率平衡误差
    add('每小时功率平衡误差<1e-4',
        e['max_balance_error'] < TOL, f"max={e['max_balance_error']:.2e}")
    # 风光出力分配误差
    add('每小时风光出力分配误差<1e-4',
        e['max_re_error'] < TOL, f"max={e['max_re_error']:.2e}")
    # 不允许电网充电（结构性保证：无 P_grid_ch 变量；P_ch=P_ch_pv+P_ch_w）
    ch_from_grid = 0.0
    for t in range(T):
        # 电网只进负荷平衡，结构性不可能充储能；校验充电功率=光伏充电+风电充电
        ch_from_grid = max(ch_from_grid,
                           abs(r_cont['P_ch'][t] - r_cont['P_ch_pv'][t] - r_cont['P_ch_w'][t]))
    add('不允许电网充电(结构性)',
        ch_from_grid < TOL, f"充电功率=光伏+风电充电，偏差={ch_from_grid:.2e}")
    # 只用余电充电（涌现验证：充电仅出现在有弃电的时段）
    emerg_ok = True
    for t in range(T):
        if r_cont['P_ch'][t] > TOL:
            surplus = G_pv_J[t] + G_w_J[t] - L_J[t]
            if surplus < -TOL:  # 缺电时段不应充电
                emerg_ok = False
                break
    add('只用余电充电(涌现)', emerg_ok,
        '充电仅出现在风光有富余的时段' if emerg_ok else '存在缺电时段充电')
    # 不同时充放电
    mutex_ok = all(not (r_cont['P_ch'][t] > TOL and r_cont['P_dis'][t] > TOL)
                   for t in range(T))
    add('不同时充放电', mutex_ok, 'z 互斥约束生效')
    # SOC 范围与日末回归（0/0 时退化，仅 E_ess>0 才实质检查）
    if r_cont['E_ess'] > 1e-6:
        add('SOC 在 10%~90%',
            e['min_soc'] >= 0.10 - TOL and e['max_soc'] <= 0.90 + TOL,
            f"min={e['min_soc']:.3f}, max={e['max_soc']:.3f}")
        add('日末SOC=日初', e['end_soc_error'] < TOL,
            f"误差={e['end_soc_error']:.2e}")
    else:
        add('SOC 范围(0/0退化)', True, 'E_ess=0，SOC 检查 vacuous')
        add('日末SOC=日初(0/0退化)', True, 'E_ess=0，SOC 检查 vacuous')
    # 年综合成本含投资
    recon = 365 * r_cont['daily_cost'] + r_cont['inv_annual']
    add('年综合成本=365×日运行+年均投资',
        abs(recon - r_cont['annual_cost']) < 1e-2,
        f"重构={recon:.3f} vs {r_cont['annual_cost']:.3f}")
    # 工程粒度
    add('工程功率符合5kW粒度',
        abs(r_int['P_ess'] - round(r_int['P_ess'] / P_STEP) * P_STEP) < TOL,
        f"P={r_int['P_ess']:.1f}")
    add('工程容量符合10kWh粒度',
        abs(r_int['E_ess'] - round(r_int['E_ess'] / E_STEP) * E_STEP) < TOL,
        f"E={r_int['E_ess']:.1f}")
    # 0/0 附近无更低成本（原点为全局最优）
    best = gs['best']
    origin_cost = next(p['annual_cost'] for p in gs['grid']
                       if p['P_ess'] == 0 and p['E_ess'] == 0)
    add('0/0 为全局最优(网格验证)',
        best['P_ess'] == 0 and best['E_ess'] == 0,
        f"网格最优={best['P_ess']:.0f}/{best['E_ess']:.0f}, 成本={best['annual_cost']:.2f}")
    bp = gs.get('best_positive')
    if bp is not None:
        add('最优正配置仍劣于0/0',
            bp['annual_cost'] > origin_cost,
            f"正配置 {bp['P_ess']:.0f}/{bp['E_ess']:.0f} 成本={bp['annual_cost']:.2f} "
            f"> 0/0={origin_cost:.2f}")
    # 充电成本敏感性：c≥0.40(光伏充电价) 时为 0；盈亏平衡点落在 (0.35, 0.40]
    zero_above = all((r['E_ess'] or 0) < 1e-3 for r in sens_c if r['c'] >= 0.40)
    add('充电价≥0.40(光伏价) 时容量为0', zero_above,
        'c≥0.40 全部 0/0' if zero_above else '存在 c≥0.40 仍为正')
    pos_below = all((r['E_ess'] or 0) > 1e-3 for r in sens_c if r['c'] <= 0.35)
    add('盈亏平衡充电价∈(0.35,0.40]', zero_above and pos_below,
        'c≤0.35 仍为正、c≥0.40 归零' if (zero_above and pos_below) else '边界与预期不符')
    # 投资价敏感性：s=1 时为 0
    base_row = next(r for r in sens_inv if r['s'] == 1.0)
    add('投资价基准(s=1)时容量为0',
        base_row['E_ess'] < 1e-3,
        f"s=1 最优={base_row['P_ess']:.0f}/{base_row['E_ess']:.0f}")
    return checks


# =====================================================================
# 主流程
# =====================================================================

def run_q2_2(output_dir=None):
    """问题2（2）完整求解流程。"""
    if output_dir is None:
        output_dir = os.path.join(_HERE, 'output')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('问题2（2）：联合园区共享储能优化')
    print('=' * 60)

    print('\n[1/8] 读取数据 + 联合 profile...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)

    print('\n[2/8] 基准：联合无储能 + 固定50kW/100kWh...')
    r_no = run_no_storage(L_J, G_pv_J, G_w_J)
    r_fixed = run_fixed_storage(L_J, G_pv_J, G_w_J, P_ess=50, E_ess=100)
    print(f"  无储能: 日成本={r_no['daily_cost']:.3f}, 年综合={r_no['annual_cost']:.3f}")
    print(f"  50/100: 日成本={r_fixed['daily_cost']:.3f}, 年综合={r_fixed['annual_cost']:.3f}, "
          f"年均投资={r_fixed['inv_annual']:.3f}")

    print('\n[3/8] 连续 MILP（理论最优）...')
    t0 = time.time()
    r_cont = solve_continuous(L_J, G_pv_J, G_w_J)
    print(f"  最优: P={r_cont['P_ess']:.4f} kW, E={r_cont['E_ess']:.4f} kWh, "
          f"年综合={r_cont['annual_cost']:.3f} ({time.time()-t0:.1f}s)")

    print('\n[4/8] 工程整数 MILP (5kW/10kWh)...')
    t0 = time.time()
    r_int = solve_integer(L_J, G_pv_J, G_w_J)
    print(f"  最优: P={r_int['P_ess']:.1f} kW, E={r_int['E_ess']:.1f} kWh, "
          f"年综合={r_int['annual_cost']:.3f} ({time.time()-t0:.1f}s)")

    print('\n[5/8] 2D 网格枚举 (P∈[0,200]/5, E∈[0,600]/10, 2501点)...')
    t0 = time.time()
    gs = grid_search_2d(L_J, G_pv_J, G_w_J, verbose=True)
    bp = gs['best_positive']
    print(f"  网格全局最优: {gs['best']['P_ess']:.0f}/{gs['best']['E_ess']:.0f}, "
          f"成本={gs['best']['annual_cost']:.3f}")
    if bp:
        print(f"  最优正配置(P>0,E>0): {bp['P_ess']:.0f}/{bp['E_ess']:.0f}, "
              f"成本={bp['annual_cost']:.3f}")
    print(f"  耗时 {time.time()-t0:.1f}s")

    print('\n[6/8] 敏感性分析...')
    sens_c = sensitivity_charge_cost(L_J, G_pv_J, G_w_J)
    print('  充电成本敏感性:')
    for r in sens_c:
        print(f"    c={r['c']:.2f}: P={r['P_ess'] or 0:.1f}, E={r['E_ess'] or 0:.1f}, "
              f"年综合={r['annual_cost'] or 0:.1f}")
    sens_inv = sensitivity_investment(gs['grid'])
    print('  投资价格敏感性:')
    for r in sens_inv:
        print(f"    s={r['s']:.2f}: P={r['P_ess']:.0f}, E={r['E_ess']:.0f}, "
              f"年综合={r['annual_cost']:.1f}")

    print('\n[7/8] 验收...')
    checks = verify_q2_2(r_cont, r_int, gs, sens_c, sens_inv, L_J, G_pv_J, G_w_J)
    n_pass = sum(1 for _, p, _ in checks if p)
    for name, passed, detail in checks:
        print(f"  {'✓' if passed else '✗'} {name}  ({detail})")
    print(f"  通过 {n_pass}/{len(checks)} 项检查")

    print('\n[8/8] 输出表格与图片...')
    df_scheme = build_scheme_table(r_no, r_fixed, r_cont, r_int,
                                   gs.get('best_positive'), L_J, G_pv_J, G_w_J)
    df_hourly, soc_row = build_hourly_table(r_cont, L_J, G_pv_J, G_w_J)
    df_sens_c = pd.DataFrame(sens_c)
    df_sens_inv = pd.DataFrame(sens_inv)

    p1 = export_result(df_scheme, os.path.join(output_dir, '问题2_2_方案对比表.xlsx'))
    p2 = export_result(df_hourly, os.path.join(output_dir, '问题2_2_逐时运行表.xlsx'))
    p3 = export_result(df_sens_c, os.path.join(output_dir, '问题2_2_充电成本敏感性.xlsx'))
    p4 = export_result(df_sens_inv, os.path.join(output_dir, '问题2_2_投资价格敏感性.xlsx'))
    for p in (p1, p2, p3, p4):
        print(f"  {p}")

    plot_heatmap(gs, os.path.join(output_dir, '问题2_2_年综合成本热力图.png'))
    plot_capacity_cost_curve(gs, os.path.join(output_dir, '问题2_2_容量成本曲线.png'))
    plot_sensitivity_charge(sens_c, os.path.join(output_dir, '问题2_2_充电成本敏感性.png'))
    plot_sensitivity_investment(sens_inv, os.path.join(output_dir, '问题2_2_投资价格敏感性.png'))
    print('  4 张图已保存')

    return {
        'r_no': r_no, 'r_fixed': r_fixed, 'r_cont': r_cont, 'r_int': r_int,
        'grid_search': gs, 'sens_c': sens_c, 'sens_inv': sens_inv,
        'checks': checks,
        'df_scheme': df_scheme, 'df_hourly': df_hourly, 'soc_row': soc_row,
        'df_sens_c': df_sens_c, 'df_sens_inv': df_sens_inv,
        'L_J': L_J, 'G_pv_J': G_pv_J, 'G_w_J': G_w_J,
    }


if __name__ == '__main__':
    run_q2_2()
    print('\nq2_storage 运行完毕')
