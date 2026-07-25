# -*- coding: utf-8 -*-
"""
问题2（2）Pareto 主模型：ε-约束法构造年实际成本-风光消纳率 Pareto 前沿（ADR 0004）。

取代 ADR 0002/0003 以 0/0 为 Q2(2) 主结论的框架：纯经济最优为 0 kW/0 kWh（弃电免费时
储能投资无消纳收益可覆盖），故采用 ε-约束法在"年实际成本最低"目标下逐档设定最低风光
消纳率 R0，求各档最优共享储能配置，构造 Pareto 前沿，再用等权归一化理想点距离选折中点。

模型（与 q2_storage_penalty.py 的 λ 法同约束集，仅目标/选型不同）：
    目标      min C_ann = 365·C_day + C_inv,ann
              C_day = Σ_t [0.4(P_pv,L+P_pv,ch) + 0.5(P_w,L+P_w,ch) + P_grid] Δt   （不含弃电惩罚）
    ε-约束   R_re = 1 − E_curt/E_re ≥ R0  ⇔  Σ_t(P_curt,pv+P_curt,w)·Δt ≤ (1−R0)·E_re
    R0       ∈ {95%, 97%, 98%, 99%, 100%}
    选型      候选 = {纯经济基准 0/0} ∪ {各 R0 连续最优}；等权理想点距离 d=√(Ĉ²+Ĝ²) 最小者
              为折中目标，再取该目标的工程整数 (5kW/10kWh) 方案为推荐配置。

λ 法（q2_storage_penalty.py）保留为补充敏感性分析，代码与输出不动，与本文件互不耦合。

运行：
    $env:PYTHONUTF8='1'; & <math-python> Q2/q2_pareto.py
    & <math-python> -m pytest Q2/test_q2_pareto.py -v
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
    load_load_data, load_solar_wind_data, DT,
)
from milp_model import (  # noqa: E402
    solve_park_optimization, run_engineering_storage, run_no_storage,
)
from q2_main import build_joint_profile, _fix_chinese_font  # noqa: E402
from common.io_utils import export_result  # noqa: E402
from common.plot_style import set_chinese_style  # noqa: E402

T = 24
# 工程粒度与物理充分上界（与 q2_storage_penalty.py 一致，详见 ADR 0003）：
# 最大逐时风光余电 448.115 kW -> 5 kW 取整 450 kW；
# 全日余电 1237.175 kWh 经 ηc=0.95 与 0.8·E 区间折算 1469.145 kWh -> 10 kWh 取整 1470 kWh。
P_STEP, E_STEP = 5, 10
P_MAX, E_MAX = 450, 1470
SURPLUS_ONLY_CHARGE = True
TOL = 1e-4

# ε-约束的最低消纳率档位
R0_TARGETS = [0.95, 0.97, 0.98, 0.99, 1.00]

# 联合无储能基准（Q2.1 结论，用于"较基准增加"列与口径锚定）
JOINT_NO_STORAGE_ANNUAL = 5514296.083
JOINT_NO_STORAGE_RE_RATIO = 0.9243683711331006


# =====================================================================
# 求解
# =====================================================================

def solve_continuous_at(L_J, G_pv_J, G_w_J, R0=None):
    """连续 MILP：给定最低消纳率 R0 的理论最优。R0=None 时不加约束（纯经济基准 0/0）。"""
    kw = dict(optimize_capacity=True, capacity_bounds=(P_MAX, E_MAX),
              surplus_only_charge=SURPLUS_ONLY_CHARGE)
    if R0 is not None:
        kw['min_accom_rate'] = R0
    return solve_park_optimization(L_J, G_pv_J, G_w_J, **kw)


def solve_integer_at(L_J, G_pv_J, G_w_J, R0=None, P_max=P_MAX, E_max=E_MAX):
    """工程整数 MILP（5kW/10kWh 粒度）：给定 R0 的工程最优。R0=None 时不加约束。"""
    kw = dict(P_step=P_STEP, E_step=E_STEP, P_max=P_max, E_max=E_max,
              surplus_only_charge=SURPLUS_ONLY_CHARGE)
    if R0 is not None:
        kw['min_accom_rate'] = R0
    return run_engineering_storage(L_J, G_pv_J, G_w_J, **kw)


def build_pareto_front(L_J, G_pv_J, G_w_J, verbose=True):
    """对每个 R0 求连续 MILP + 工程整数 MILP，并附纯经济基准。

    返回 dict:
        'baseline_cont' : 纯经济连续最优（0/0）
        'baseline_int'  : 纯经济工程整数最优（0/0）
        'by_R0'         : {R0: {'cont': r_cont, 'int': r_int}}
    """
    print('  [a] 纯经济基准（无消纳约束）...')
    t0 = time.time()
    r_base_c = solve_continuous_at(L_J, G_pv_J, G_w_J, R0=None)
    r_base_i = solve_integer_at(L_J, G_pv_J, G_w_J, R0=None)
    if verbose:
        print(f"      连续: P={r_base_c['P_ess']:.3f}, E={r_base_c['E_ess']:.3f}, "
              f"年实际={r_base_c['annual_cost']:.2f}, 消纳率={r_base_c['re_ratio']:.6f} "
              f"({time.time()-t0:.1f}s)")

    by_R0 = {}
    for R0 in R0_TARGETS:
        t0 = time.time()
        r_c = solve_continuous_at(L_J, G_pv_J, G_w_J, R0=R0)
        r_i = solve_integer_at(L_J, G_pv_J, G_w_J, R0=R0)
        by_R0[R0] = {'cont': r_c, 'int': r_i}
        if verbose:
            print(f"  [R0={R0*100:.0f}%] 连续: P={r_c['P_ess']:.3f}, E={r_c['E_ess']:.3f}, "
                  f"年实际={r_c['annual_cost']:.2f}, 消纳率={r_c['re_ratio']:.6f}")
            print(f"           工程: P={r_i['P_ess']:.0f}, E={r_i['E_ess']:.0f}, "
                  f"年实际={r_i['annual_cost']:.2f}, 消纳率={r_i['re_ratio']:.6f} "
                  f"({time.time()-t0:.1f}s)")

    return {
        'baseline_cont': r_base_c,
        'baseline_int': r_base_i,
        'by_R0': by_R0,
    }


# =====================================================================
# 等权理想点距离选型
# =====================================================================

def ideal_point_selection(front):
    """候选集 = {纯经济基准(连续)} ∪ {各 R0 连续最优}。

    归一化（成本越小越好、消纳率越高越好）：
        Ĉ_j = (C_j − C_min) / (C_max − C_min)
        Ĝ_j = (R_max − R_j) / (R_max − R_min)      # 绿色缺口，越大越差
        d_j = √(Ĉ_j² + Ĝ_j²)
    选 d_j 最小的目标 R0*，再取该 R0 的工程整数方案为推荐配置。
    """
    candidates = [{
        'label': '纯经济基准', 'R0': None,
        're_ratio': front['baseline_cont']['re_ratio'],
        'annual_cost': front['baseline_cont']['annual_cost'],
        'P': front['baseline_cont']['P_ess'],
        'E': front['baseline_cont']['E_ess'],
    }]
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['cont']
        candidates.append({
            'label': f'{R0*100:.0f}%目标', 'R0': R0,
            're_ratio': r['re_ratio'],
            'annual_cost': r['annual_cost'],
            'P': r['P_ess'], 'E': r['E_ess'],
        })

    costs = [c['annual_cost'] for c in candidates]
    ratios = [c['re_ratio'] for c in candidates]
    C_min, C_max = min(costs), max(costs)
    R_min, R_max = min(ratios), max(ratios)
    c_span = C_max - C_min if C_max > C_min else 1.0
    r_span = R_max - R_min if R_max > R_min else 1.0

    for c in candidates:
        c['cost_norm'] = (c['annual_cost'] - C_min) / c_span
        c['green_gap'] = (R_max - c['re_ratio']) / r_span
        c['ideal_dist'] = float(np.sqrt(c['cost_norm']**2 + c['green_gap']**2))

    best = min(candidates, key=lambda c: c['ideal_dist'])
    # 推荐配置 = 最佳目标 R0* 的工程整数方案（基准若被选则工程整数亦为 0/0）
    if best['R0'] is None:
        rec_R0, rec_int = None, front['baseline_int']
    else:
        rec_R0, rec_int = best['R0'], front['by_R0'][best['R0']]['int']

    return {
        'candidates': candidates,
        'best': best,
        'recommended_R0': rec_R0,
        'recommended_int': rec_int,
        'ranges': {'C_min': C_min, 'C_max': C_max, 'R_min': R_min, 'R_max': R_max},
    }


# =====================================================================
# 表格构造
# =====================================================================

def build_continuous_table(front):
    """§2.2.9 连续理论最优表。"""
    rows = []
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['cont']
        rows.append({
            '最低消纳率': f'{R0*100:.0f}%',
            '功率/kW': round(r['P_ess'], 3),
            '容量/kWh': round(r['E_ess'], 3),
            'E/P/h': round(r['E_ess'] / r['P_ess'], 3) if r['P_ess'] > 1e-6 else 0.0,
            '日弃电/kWh': round(r['curt_total'], 3),
            '年实际成本/元': round(r['annual_cost'], 2),
        })
    return pd.DataFrame(rows)


def build_integer_table(front):
    """§2.2.9 工程整数最优表（含实际消纳率与较基准增加）。"""
    base = front['baseline_int']['annual_cost']
    rows = []
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        rows.append({
            '最低消纳率': f'{R0*100:.0f}%',
            '功率/kW': int(round(r['P_ess'])),
            '容量/kWh': int(round(r['E_ess'])),
            'E/P/h': round(r['E_ess'] / r['P_ess'], 3) if r['P_ess'] > 1e-6 else 0.0,
            '日弃电/kWh': round(r['curt_total'], 3),
            '实际消纳率': f"{r['re_ratio']*100:.3f}%",
            '年实际成本/元': round(r['annual_cost'], 2),
            '较基准增加/元': round(r['annual_cost'] - base, 2),
        })
    return pd.DataFrame(rows)


def build_ideal_point_table(selection):
    """§2.2.9 理想点距离表。"""
    rows = []
    for c in selection['candidates']:
        rows.append({
            '候选方案': c['label'],
            '消纳率': f"{c['re_ratio']*100:.3f}%",
            '年实际成本/元': round(c['annual_cost'], 2),
            '理想点距离': round(c['ideal_dist'], 3),
        })
    return pd.DataFrame(rows)


def build_hourly_table(r_opt, L_J, G_pv_J, G_w_J):
    """§2.2.10 推荐方案逐时调度表。储能电量为时段初值 E[t]（t=0..23）。"""
    soc_start = list(r_opt['E'])[:T]  # E[0..23]，时段初电量
    direct = [r_opt['P_load_pv'][t] + r_opt['P_load_w'][t] for t in range(T)]
    curt = [r_opt['P_curt_pv'][t] + r_opt['P_curt_w'][t] for t in range(T)]
    df = pd.DataFrame({
        '时刻': list(range(T)),
        '联合负荷/kW': [round(float(x), 3) for x in L_J],
        '光伏出力/kW': [round(float(x), 3) for x in G_pv_J],
        '风电出力/kW': [round(float(x), 3) for x in G_w_J],
        '风光直供/kW': [round(x, 3) for x in direct],
        '储能充电/kW': [round(float(x), 3) for x in r_opt['P_ch']],
        '储能放电/kW': [round(float(x), 3) for x in r_opt['P_dis']],
        '电网购电/kW': [round(float(x), 3) for x in r_opt['P_grid']],
        '弃电/kW': [round(x, 3) for x in curt],
        '储能电量/kWh': [round(float(x), 3) for x in soc_start],
    })
    return df


def build_economic_summary(r_opt, L_J):
    """§2.2.11 推荐方案经济性汇总。"""
    load_total = float(np.sum(L_J) * DT)
    rows = [
        ('日实际运行成本/元', round(r_opt['daily_cost'], 3)),
        ('储能年化投资/元', round(r_opt['inv_annual'], 2)),
        ('年实际成本/元', round(r_opt['annual_cost'], 2)),
        ('单位负荷电量年实际成本/(元·kWh⁻¹)',
         round(r_opt['annual_cost'] / (365 * load_total), 4)),
        ('日电网购电量/kWh', round(r_opt['grid_total'], 3)),
        ('日弃电量/kWh', round(r_opt['curt_total'], 3)),
        ('风光消纳率', f"{r_opt['re_ratio']*100:.3f}%"),
        ('储能功率/kW', int(round(r_opt['P_ess']))),
        ('储能容量/kWh', int(round(r_opt['E_ess']))),
    ]
    return pd.DataFrame(rows, columns=['指标', '结果'])


# =====================================================================
# 绘图
# =====================================================================

def plot_pareto_front(selection, save_path=None):
    """Pareto 前沿：年实际成本 vs 风光消纳率，标注推荐折中点。"""
    set_chinese_style()
    _fix_chinese_font()

    fig, ax = plt.subplots(figsize=(8, 5))
    cands = selection['candidates']
    xs = [c['re_ratio'] * 100 for c in cands]
    ys = [c['annual_cost'] / 1e4 for c in cands]
    ax.plot(xs, ys, '-o', color='#2c7fb8', linewidth=1.5, markersize=7, zorder=2)
    for c in cands:
        ax.annotate(c['label'], (c['re_ratio']*100, c['annual_cost']/1e4),
                    textcoords='offset points', xytext=(6, 6), fontsize=9)
    rec = selection['best']
    ax.scatter([rec['re_ratio']*100], [rec['annual_cost']/1e4],
               s=160, marker='*', color='#d95f0e', zorder=3,
               label=f"推荐折中点 {rec['label']}")
    ax.set_xlabel('风光消纳率 / %')
    ax.set_ylabel('年实际成本 / 万元')
    ax.set_title('问题2.2 Pareto 前沿（年实际成本-风光消纳率）')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 验证（供 test_q2_pareto.py 调用）
# =====================================================================

def verify_pareto(front, selection, L_J, G_pv_J, G_w_J):
    """§2.2.12 的 9 项自动化检查。返回 [(name, ok, detail), ...]。"""
    checks = []

    # 1. 所有方案均满足设定的消纳率
    ok = True
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        if r['re_ratio'] < R0 - 1e-6:
            ok = False
            break
    checks.append(('1.各方案满足设定消纳率', ok,
                   f"工程整数各档 re_ratio≥R0" if ok else f"R0={R0} 未满足"))

    # 2. 消纳目标越严格，弃电量不增加、最优成本不降低
    curts = [front['by_R0'][R0]['int']['curt_total'] for R0 in R0_TARGETS]
    costs = [front['by_R0'][R0]['int']['annual_cost'] for R0 in R0_TARGETS]
    curt_nonincr = all(curts[i+1] <= curts[i] + 1e-6 for i in range(len(curts)-1))
    cost_nondecr = all(costs[i+1] >= costs[i] - 1e-3 for i in range(len(costs)-1))
    checks.append(('2.目标越严弃电不增、成本不降', curt_nonincr and cost_nondecr,
                   f"弃电序列单调非增={curt_nonincr}, 成本单调非降={cost_nondecr}"))

    # 3. 工程容量满足 5kW/10kWh 粒度
    ok = all((abs(front['by_R0'][R0]['int']['P_ess'] % P_STEP) < 1e-6) and
             (abs(front['by_R0'][R0]['int']['E_ess'] % E_STEP) < 1e-6)
             for R0 in R0_TARGETS)
    checks.append(('3.工程容量5kW/10kWh粒度', ok, '各档 P%5=0、E%10=0' if ok else '粒度不符'))

    # 4. 功率平衡和风光分配最大误差不超过 1e-5
    max_err = 0.0
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        max_err = max(max_err, r['errors']['max_balance_error'],
                      r['errors']['max_re_error'])
    checks.append(('4.功率平衡/风光分配误差≤1e-5', max_err <= 1e-5,
                   f"max_err={max_err:.2e}"))

    # 5. 储能只使用风光余电充电
    ok = True
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        for t in range(T):
            surplus = max(float(G_pv_J[t]) + float(G_w_J[t]) - float(L_J[t]), 0.0)
            if r['P_ch'][t] > surplus + 1e-6:
                ok = False
                break
        if not ok:
            break
    checks.append(('5.只用风光余电充电', ok, '逐时 P_ch≤max(G_pv+G_w−L,0)' if ok else '违规充电'))

    # 6. 充放电互斥、SOC范围、初末SOC
    ok = True
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        E_cap = r['E_ess']
        if E_cap > 1e-6:
            soc = [r['E'][t] / E_cap for t in range(T + 1)]
            if min(soc) < 0.1 - 1e-6 or max(soc) > 0.9 + 1e-6:
                ok = False
                break
        # 互斥：z_t=1 时 P_dis=0，z_t=0 时 P_ch=0
        for t in range(T):
            if r['z'][t] is not None:
                zt = r['z'][t]
                if zt > 0.5 and r['P_dis'][t] > 1e-6:
                    ok = False
                    break
                if zt < 0.5 and r['P_ch'][t] > 1e-6:
                    ok = False
                    break
        if abs(r['E'][T] - r['E'][0]) > 1e-4:
            ok = False
            break
    checks.append(('6.充放互斥/SOC区间/初末SOC', ok, '全部满足' if ok else '存在违反'))

    # 7. 消纳率约束与弃电率约束等价（re_ratio≥R0 ⇔ E_curt≤(1−R0)·E_re）
    ok = True
    re_gen = float(np.sum(G_pv_J) + np.sum(G_w_J)) * DT
    for R0 in R0_TARGETS:
        r = front['by_R0'][R0]['int']
        curt = r['curt_total']
        re_ratio = 1 - curt / re_gen
        if abs(re_ratio - r['re_ratio']) > 1e-6:
            ok = False
            break
        if curt > (1 - R0) * re_gen + 1e-6:
            ok = False
            break
    checks.append(('7.消纳率与弃电率约束等价', ok, '两种表述同解' if ok else '不一致'))

    # 8. 零弃电方案实现 100% 消纳
    r100 = front['by_R0'][1.00]['int']
    ok = (abs(r100['curt_total']) < 1e-6) and (abs(r100['re_ratio'] - 1.0) < 1e-6)
    checks.append(('8.零弃电方案100%消纳', ok,
                   f"R0=100%: 弃电={r100['curt_total']:.4f}, 消纳率={r100['re_ratio']:.6f}"))

    # 9. 容量上界放宽到 550/1800 后，零弃电工程最优仍为 450/1470
    r100_wide = solve_integer_at(L_J, G_pv_J, G_w_J, R0=1.00, P_max=550, E_max=1800)
    ok = (abs(r100_wide['P_ess'] - 450) < 1e-6) and (abs(r100_wide['E_ess'] - 1470) < 1e-6)
    checks.append(('9.放宽上界零弃电仍为450/1470', ok,
                   f"放宽后: {r100_wide['P_ess']:.0f}/{r100_wide['E_ess']:.0f}"))

    return checks


# =====================================================================
# 主流程
# =====================================================================

def run_q2_pareto(output_dir=None):
    """问题2.2 Pareto 主模型完整求解流程。"""
    if output_dir is None:
        output_dir = os.path.join(_HERE, 'output_pareto')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 64)
    print('问题2（2）Pareto 主模型（ε-约束法, ADR 0004）')
    print('=' * 64)

    print('\n[1] 读取数据 + 联合 profile...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)

    print('\n[2] 构造 Pareto 前沿（连续 MILP + 工程整数 MILP，R0∈{95,97,98,99,100}%）...')
    front = build_pareto_front(L_J, G_pv_J, G_w_J)

    print('\n[3] 等权理想点距离选型...')
    sel = ideal_point_selection(front)
    rec_R0 = sel['recommended_R0']
    rec = sel['recommended_int']
    print(f"  推荐折中目标: {rec_R0*100 if rec_R0 else '基准(0/0)'}%  "
          f"理想点距离 d={sel['best']['ideal_dist']:.4f}")
    print(f"  推荐工程配置: P={rec['P_ess']:.0f} kW, E={rec['E_ess']:.0f} kWh, "
          f"实际消纳率={rec['re_ratio']*100:.3f}%, 年实际成本={rec['annual_cost']:.2f} 元")

    print('\n[4] 验证（§2.2.12 九项检查）...')
    checks = verify_pareto(front, sel, L_J, G_pv_J, G_w_J)
    n_pass = sum(1 for _, ok, _ in checks if ok)
    for name, ok, det in checks:
        print(f"  {'✓' if ok else '✗'} {name}  ({det})")
    print(f"  通过 {n_pass}/{len(checks)} 项检查")

    print('\n[5] 输出表格与图片到 output_pareto/ ...')
    df_cont = build_continuous_table(front)
    df_int = build_integer_table(front)
    df_ideal = build_ideal_point_table(sel)
    df_hourly = build_hourly_table(rec, L_J, G_pv_J, G_w_J)
    df_econ = build_economic_summary(rec, L_J)

    export_result(df_cont, os.path.join(output_dir, 'Pareto连续理论最优.xlsx'))
    export_result(df_int, os.path.join(output_dir, 'Pareto工程整数最优.xlsx'))
    export_result(df_ideal, os.path.join(output_dir, '理想点距离表.xlsx'))
    export_result(df_hourly, os.path.join(output_dir, '推荐方案逐时调度表.xlsx'))
    export_result(df_econ, os.path.join(output_dir, '推荐方案经济汇总.xlsx'))
    plot_pareto_front(sel, os.path.join(output_dir, 'Pareto前沿曲线.png'))

    # 控制台打印关键结论
    print('\n' + '=' * 64)
    print('Pareto 前沿（连续）:')
    print(df_cont.to_string(index=False))
    print('\n理想点距离:')
    print(df_ideal.to_string(index=False))
    base = front['baseline_int']['annual_cost']
    print(f"\n推荐方案: P={rec['P_ess']:.0f} kW, E={rec['E_ess']:.0f} kWh")
    print(f"  实际消纳率={rec['re_ratio']*100:.3f}%, 年实际成本={rec['annual_cost']:.2f} 元, "
          f"较0/0基准增加 {rec['annual_cost']-base:.2f} 元")
    print('=' * 64)

    return {'front': front, 'selection': sel, 'checks': checks,
            'recommended': rec, 'profile': (L_J, G_pv_J, G_w_J)}


if __name__ == '__main__':
    run_q2_pareto()
