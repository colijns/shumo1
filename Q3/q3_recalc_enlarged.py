# -*- coding: utf-8 -*-
"""
问题3（1）扩界复算脚本：将容量上界 1.5× 后重跑完整 Pareto 扫描，
确认 94.22% 仍为折中优选。

基于 docs/问题3（1）Pareto运行结果审查.md 第1项待办：
  独立 98.19% 方案新增光伏达到 4000 kW 上界，需扩大后复算。

用法：
    $env:PYTHONUTF8='1'; & <math-python> Q3/q3_recalc_enlarged.py
    $env:PYTHONUTF8='1'; & <math-python> Q3/q3_recalc_enlarged.py --no-integers  # 仅连续，快速验证
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'templates'))
for _d in (_HERE, _Q1_DIR, _TEMPLATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import q3_model as M  # noqa: E402
from q3_model import prepare_data, solve_independent, solve_joint, PARK_LIST, T  # noqa: E402

# ---- 复用 q3_pareto 的工具函数 ----
import q3_pareto as P  # noqa: E402

# =====================================================================
# 参数
# =====================================================================
# 旧值（用于对比）
OLD_DK_MAX = M.DK_MAX
OLD_P_MAX_IND = M.P_MAX_IND
OLD_E_MAX_IND = M.E_MAX_IND
OLD_P_MAX_JNT = M.P_MAX_JNT
OLD_E_MAX_JNT = M.E_MAX_JNT

# 1.5× 新值
NEW_DK_MAX = int(OLD_DK_MAX * 1.5)
NEW_P_MAX_IND = int(OLD_P_MAX_IND * 1.5)
NEW_E_MAX_IND = int(OLD_E_MAX_IND * 1.5)
NEW_P_MAX_JNT = int(OLD_P_MAX_JNT * 1.5)
NEW_E_MAX_JNT = int(OLD_E_MAX_JNT * 1.5)

OLD_COMPROMISE = 0.9422  # 旧折中 R0

OUTPUT_DIR = os.path.join(_HERE, 'output_pareto_enlarged')

# =====================================================================
# 按园区拆分输出
# =====================================================================
def build_per_park_independent(r, data):
    """独立运营：按园区拆分风光储配置。"""
    K0pv = data['K0_pv']
    K0w = data['K0_w']
    caps = r['capacities']
    rows = []
    for p in PARK_LIST:
        dKpv = caps[p]['dK_pv']
        dKw = caps[p]['dK_w']
        Pess = caps[p]['P_ess']
        Eess = caps[p]['E_ess']
        total_pv = K0pv[p] + dKpv
        total_w = K0w[p] + dKw
        rows.append({
            '园区': p,
            '现有光伏/kW': K0pv[p],
            '现有风电/kW': K0w[p],
            'Δ光伏/kW': round(dKpv, 1),
            'Δ风电/kW': round(dKw, 1),
            '储能功率/kW': round(Pess, 1),
            '储能容量/kWh': round(Eess, 1),
            'E/P/h': round(Eess / Pess, 3) if Pess > 1e-6 else 0.0,
            '合计光伏/kW': round(total_pv, 1),
            '合计风电/kW': round(total_w, 1),
        })
    # 合计行
    sum_dKpv = sum(caps[p]['dK_pv'] for p in PARK_LIST)
    sum_dKw = sum(caps[p]['dK_w'] for p in PARK_LIST)
    sum_P = sum(caps[p]['P_ess'] for p in PARK_LIST)
    sum_E = sum(caps[p]['E_ess'] for p in PARK_LIST)
    rows.append({
        '园区': '合计',
        '现有光伏/kW': sum(K0pv[p] for p in PARK_LIST),
        '现有风电/kW': sum(K0w[p] for p in PARK_LIST),
        'Δ光伏/kW': round(sum_dKpv, 1),
        'Δ风电/kW': round(sum_dKw, 1),
        '储能功率/kW': round(sum_P, 1),
        '储能容量/kWh': round(sum_E, 1),
        'E/P/h': round(sum_E / sum_P, 3) if sum_P > 1e-6 else 0.0,
        '合计光伏/kW': round(sum(K0pv[p] for p in PARK_LIST) + sum_dKpv, 1),
        '合计风电/kW': round(sum(K0w[p] for p in PARK_LIST) + sum_dKw, 1),
    })
    return pd.DataFrame(rows)


def build_per_park_joint(r, data):
    """联合运营：新增风光设备归属园区 + 共享储能。"""
    K0pv = data['K0_pv']
    K0w = data['K0_w']
    caps = r['capacities']
    rows = []
    for p in PARK_LIST:
        dKpv = caps[p]['dK_pv']
        dKw = caps[p]['dK_w']
        total_pv = K0pv[p] + dKpv
        total_w = K0w[p] + dKw
        rows.append({
            '园区': p,
            '现有光伏/kW': K0pv[p],
            '现有风电/kW': K0w[p],
            'Δ光伏/kW': round(dKpv, 1),
            'Δ风电/kW': round(dKw, 1),
            '合计光伏/kW': round(total_pv, 1),
            '合计风电/kW': round(total_w, 1),
        })
    # 合计行
    sum_dKpv = sum(caps[p]['dK_pv'] for p in PARK_LIST)
    sum_dKw = sum(caps[p]['dK_w'] for p in PARK_LIST)
    rows.append({
        '园区': '合计',
        '现有光伏/kW': sum(K0pv[p] for p in PARK_LIST),
        '现有风电/kW': sum(K0w[p] for p in PARK_LIST),
        'Δ光伏/kW': round(sum_dKpv, 1),
        'Δ风电/kW': round(sum_dKw, 1),
        '合计光伏/kW': round(sum(K0pv[p] for p in PARK_LIST) + sum_dKpv, 1),
        '合计风电/kW': round(sum(K0w[p] for p in PARK_LIST) + sum_dKw, 1),
    })
    return pd.DataFrame(rows)


def build_joint_storage_info(r):
    """联合运营共享储能信息（单行 DataFrame）。"""
    PJ = r.get('P_J', 0)
    EJ = r.get('E_J', 0)
    return pd.DataFrame([{
        '储能功率/kW': round(PJ, 1),
        '储能容量/kWh': round(EJ, 1),
        'E/P/h': round(EJ / PJ, 3) if PJ > 1e-6 else 0.0,
    }])


# =====================================================================

# =====================================================================
# 逐园区校验与逐时调度
# =====================================================================
def validate_per_park(r, data):
    """逐园区校验 5 项：充放电互斥、禁电网充电、SOC区间、功率平衡、初末SOC。"""
    T = 24
    DT = 1.0
    hourly = r['hourly']
    load = data['load']
    rows = []
    all_ok = True
    for p in PARK_LIST:
        h = hourly[p]
        ch = np.array(h['ch'])
        dis = np.array(h['dis'])
        grid = np.array(h['grid'])
        e_vals = np.array(h['E'])
        soc = e_vals / max(r['capacities'][p]['E_ess'], 1e-6)

        # 1. 同时充放电
        vio1 = np.any((ch > 1e-3) & (dis > 1e-3))
        # 2. 储能充自电网
        vio2 = np.any((ch > 1e-3) & (grid > 1e-3))
        # 3. SOC 区间 [0.10, 0.90]
        soc_lo = np.min(soc[:T])  # only hours 0..23
        soc_hi = np.max(soc[:T])
        vio3 = (soc_lo < 0.10 - 1e-4) or (soc_hi > 0.90 + 1e-4)
        # 4. 功率平衡
        load_pv = np.array(h.get('load_pv', np.zeros(T)))
        load_w = np.array(h.get('load_w', np.zeros(T)))
        balance = load_pv + load_w + dis + grid - np.array(load[p])
        max_balance_err = np.max(np.abs(balance))
        vio4 = max_balance_err > 1e-3
        # 5. SOC 初末相等
        soc_end_err = abs(soc[0] - soc[T])
        vio5 = soc_end_err > 1e-3

        ok_count = sum(1 for v in [vio1, vio2, vio3, vio4, vio5] if not v)
        if ok_count < 5:
            all_ok = False

        rows.append({
            '园区': p,
            '同时充放电': '✗' if vio1 else '✓',
            '储能充自电网': '✗' if vio2 else '✓',
            'SOC区间 [0.10,0.90]': f'{soc_lo:.4f}~{soc_hi:.4f} {"✗" if vio3 else "✓"}',
            '功率平衡 max|err|': f'{max_balance_err:.2e} {"✗" if vio4 else "✓"}',
            'SOC初末差': f'{soc_end_err:.2e} {"✗" if vio5 else "✓"}',
        })
    df = pd.DataFrame(rows)
    return df, all_ok


def build_hourly_per_park(r, park_id):
    """单园区逐时调度明细表。"""
    h = r['hourly'][park_id]
    T = 24
    rows = []
    for t in range(T):
        rows.append({
            '时刻': t,
            '负荷/kW': round(h['load_pv'][t] + h['load_w'][t] + h['dis'][t] + h['grid'][t], 2),
            '风光直供/kW': round(h['load_pv'][t] + h['load_w'][t], 2),
            '储能充电/kW': round(h['ch'][t], 2),
            '储能放电/kW': round(h['dis'][t], 2),
            '电网购电/kW': round(h['grid'][t], 2),
            '弃光/kW': round(h.get('curt_pv', [0]*T)[t], 2),
            '弃风/kW': round(h.get('curt_w', [0]*T)[t], 2),
            '储能电量/kWh': round(h['E'][t], 2),
        })
    return pd.DataFrame(rows)


def plot_schedule_per_park(r, park_id, save_path=None):
    """单园区逐时调度堆叠图。"""
    P._fix_chinese_font()
    P.set_chinese_style()
    df = build_hourly_per_park(r, park_id)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    t = df['时刻']
    ax1.fill_between(t, 0, df['风光直供/kW'], color='#74c476', alpha=0.7, label='风光直供')
    ax1.bar(t, df['储能放电/kW'], bottom=df['风光直供/kW'], color='#fd8d3c', alpha=0.7, label='储能放电')
    ax1.bar(t, -df['储能充电/kW'], color='#9e9ac8', alpha=0.7, label='储能充电')
    ax1.plot(t, df['电网购电/kW'], color='#de2d26', linewidth=1.5, label='电网购电')
    ax1.set_xlabel('时刻 / h'); ax1.set_ylabel('功率 / kW')
    ax1.set_title(f"问题3（1）推荐方案逐时调度 — 园区{park_id}（{r['mode']}）")
    ax2 = ax1.twinx()
    ax2.plot(t, df['储能电量/kWh'], color='#3182bd', linewidth=1.5, linestyle='--', label='储能电量')
    ax2.set_ylabel('储能电量 / kWh')
    ax1.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax1.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# 主流程
# =====================================================================
def run():
    """扩界复算主流程。"""
    print('=' * 70)
    print('问题3（1）扩界复算（1.5× 上界）')
    print('=' * 70)

    # --- 检查输出目录是否有旧结果，用于对比 ---
    old_results = _load_old_results()

    # --- Monkey-patch 上界 ---
    print(f'\n上界扩展:')
    print(f'  DK_MAX:    {OLD_DK_MAX} → {NEW_DK_MAX}')
    print(f'  P_MAX_IND: {OLD_P_MAX_IND} → {NEW_P_MAX_IND}')
    print(f'  E_MAX_IND: {OLD_E_MAX_IND} → {NEW_E_MAX_IND}')
    print(f'  P_MAX_JNT: {OLD_P_MAX_JNT} → {NEW_P_MAX_JNT}')
    print(f'  E_MAX_JNT: {OLD_E_MAX_JNT} → {NEW_E_MAX_JNT}')

    M.DK_MAX = NEW_DK_MAX
    M.P_MAX_IND = NEW_P_MAX_IND
    M.E_MAX_IND = NEW_E_MAX_IND
    M.P_MAX_JNT = NEW_P_MAX_JNT
    M.E_MAX_JNT = NEW_E_MAX_JNT

    # --- 数据 ---
    print('\n[1] 数据准备（1.5× 负荷）...')
    data = prepare_data()
    total_load = sum(np.sum(data['load'][p]) for p in PARK_LIST)
    print(f'    联合日电量={total_load:.1f} kWh')

    # --- 端点 ---
    print('\n[2] 求两端点（扩界后）...')
    ep = P.solve_endpoints(data)
    lo, hi = P.scan_range(ep)
    print(f'  独立: econ R_load={ep["ind_econ_c"]["R_load"]*100:.2f}%  '
          f'max R_load={ep["ind_max_c"]["R_load"]*100:.2f}%')
    print(f'  联合: econ R_load={ep["jnt_econ_c"]["R_load"]*100:.2f}%  '
          f'max R_load={ep["jnt_max_c"]["R_load"]*100:.2f}%')
    print(f'  扫描范围: [{lo*100:.3f}%, {hi*100:.3f}%]')

    if hi <= lo:
        print(f'  ⚠ 扫描范围为空，退出')
        return

    R0_list = list(np.linspace(lo, hi, P.R0_NPOINTS))
    print(f'  R0 目标点: {[f"{r*100:.3f}%" for r in R0_list]}')

    # --- Pareto 前沿 ---
    print('\n[3] 构造 Pareto 前沿（扩界，连续 + 工程整数）...')
    front = P.build_front(data, R0_list, integers=True)

    # --- 同时计算旧折中点 94.22%（如果在前沿范围外则单独算） ---
    old_R0 = OLD_COMPROMISE
    old_point = None
    if lo <= old_R0 <= hi and not any(abs(r - old_R0) < 1e-5 for r in R0_list):
        print(f'\n[3b] 额外计算旧折中点 R0={old_R0*100:.2f}% 以便对比...')
        old_entry = {}
        old_entry['ind_c'] = solve_independent(data, R0=old_R0, mode='cost', integer=False)
        old_entry['jnt_c'] = solve_joint(data, R0=old_R0, mode='cost', integer=False)
        old_entry['ind_i'] = solve_independent(data, R0=old_R0, mode='cost', integer=True)
        old_entry['jnt_i'] = solve_joint(data, R0=old_R0, mode='cost', integer=True)
        old_point = old_entry
        front[old_R0] = old_entry  # 加入 front 字典

    # --- 理想点选型 ---
    print('\n[4] 等权理想点距离选型（基于联合前沿）...')
    sel = P.ideal_point_select(ep, front, R0_list)
    rec_R0 = sel['best']['R0']
    print(f'  推荐折中 R0* = {rec_R0*100 if rec_R0 else "纯经济"}%  '
          f'd={sel["best"]["ideal_dist"]:.4f}')

    if rec_R0 is None:
        rec_j = ep['jnt_econ_i']
        rec_i = ep['ind_econ_i']
    else:
        rec_j = front[rec_R0]['jnt_i']
        rec_i = front[rec_R0]['ind_i']
    print(f'  推荐联合: {P._sum_caps(rec_j)}  R_load={rec_j["R_load"]*100:.3f}%  '
          f'C={rec_j["annual_cost"]:.0f}')
    print(f'  推荐独立: {P._sum_caps(rec_i)}  R_load={rec_i["R_load"]*100:.3f}%  '
          f'C={rec_i["annual_cost"]:.0f}')
    dC = rec_i['annual_cost'] - rec_j['annual_cost']
    print(f'  ΔC(独立-联合)={dC:.0f} 元  η_C={dC/rec_i["annual_cost"]*100:.3f}%')

    # --- 验证 ---
    print('\n[5] 验证...')
    checks = P.verify(ep, front, R0_list, integers=True)
    for name, ok, det in checks:
        print(f'  {"✓" if ok else "✗"} {name}  ({det})')
    print(f'  通过 {sum(1 for _,ok,_ in checks if ok)}/{len(checks)}')

    # --- 输出 ---
    print(f'\n[6] 输出表格与图片到 {OUTPUT_DIR}/ ...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_cont = P.build_continuous_table(ep, front, R0_list)
    P.export_result(df_cont, os.path.join(OUTPUT_DIR, 'Pareto连续理论最优.xlsx'))
    df_int = P.build_integer_table(ep, front, R0_list)
    P.export_result(df_int, os.path.join(OUTPUT_DIR, 'Pareto工程整数最优.xlsx'))
    df_ideal = P.build_ideal_table(sel)
    P.export_result(df_ideal, os.path.join(OUTPUT_DIR, '理想点距离表.xlsx'))
    df_cmp = P.build_compare_table(ep, front, R0_list, suffix='i')
    P.export_result(df_cmp, os.path.join(OUTPUT_DIR, '独立vs联合对比.xlsx'))
    df_marg = P.build_marginal_table(front, R0_list, 'jnt_c')
    P.export_result(df_marg, os.path.join(OUTPUT_DIR, '边际成本表.xlsx'))
    P.export_result(P.build_economic_summary(rec_j),
                    os.path.join(OUTPUT_DIR, '推荐方案经济汇总_联合.xlsx'))
    P.export_result(P.build_economic_summary(rec_i),
                    os.path.join(OUTPUT_DIR, '推荐方案经济汇总_独立.xlsx'))
    P.export_result(P.build_hourly_table(rec_j),
                    os.path.join(OUTPUT_DIR, '推荐方案逐时调度_联合.xlsx'))
    P.export_result(P.build_hourly_table(rec_i),
                    os.path.join(OUTPUT_DIR, '推荐方案逐时调度_独立.xlsx'))

    # 按园区拆分的容量配置
    df_per_park_ind = build_per_park_independent(rec_i, data)
    P.export_result(df_per_park_ind, os.path.join(OUTPUT_DIR, '推荐方案_独立_按园区拆分.xlsx'))
    df_per_park_jnt = build_per_park_joint(rec_j, data)
    P.export_result(df_per_park_jnt, os.path.join(OUTPUT_DIR, '推荐方案_联合_按园区拆分.xlsx'))
    df_ess_jnt = build_joint_storage_info(rec_j)
    P.export_result(df_ess_jnt, os.path.join(OUTPUT_DIR, '推荐方案_联合_共享储能.xlsx'))

    # 打印 per-park 表格
    print('\n--- 独立运营按园区拆分 ---')
    print(df_per_park_ind.to_string(index=False))
    print(f'\n(独立合计: ΔPV={P._sum_caps(rec_i)[0]:.0f}kW, ΔWind={P._sum_caps(rec_i)[1]:.0f}kW, '
          f'P_ess={P._sum_caps(rec_i)[2]:.0f}kW, E_ess={P._sum_caps(rec_i)[3]:.0f}kWh)')
    print('\n--- 联合运营新增设备归属 ---')
    print(df_per_park_jnt.to_string(index=False))
    print(f'\n共享储能: P_J={rec_j.get("P_J", 0):.0f} kW, E_J={rec_j.get("E_J", 0):.0f} kWh, '
          f'E/P={rec_j.get("E_J", 0) / max(rec_j.get("P_J", 1), 1e-6):.3f} h')

    # 逐园区校验
    print('\n--- 独立运营逐园区校验 ---')
    df_val, all_ok = validate_per_park(rec_i, data)
    print(df_val.to_string(index=False))
    print(f'  全部通过: {"是" if all_ok else "否 — 有违规项!"}')
    P.export_result(df_val, os.path.join(OUTPUT_DIR, '推荐方案逐时校验_独立.xlsx'))

    # 逐园区逐时调度表 + 图
    for p in PARK_LIST:
        df_hourly_p = build_hourly_per_park(rec_i, p)
        P.export_result(df_hourly_p, os.path.join(OUTPUT_DIR, f'推荐方案逐时调度_独立_{p}.xlsx'))
        plot_schedule_per_park(rec_i, p, os.path.join(OUTPUT_DIR, f'推荐方案逐时调度_独立_{p}.png'))

    P.plot_pareto(ep, front, R0_list, sel, os.path.join(OUTPUT_DIR, 'Pareto前沿曲线.png'))
    P.plot_schedule(rec_j, os.path.join(OUTPUT_DIR, '推荐方案逐时调度_联合.png'))
    P.plot_schedule(rec_i, os.path.join(OUTPUT_DIR, '推荐方案逐时调度_独立.png'))

    # --- 扩界前后对比 ---
    print('\n' + '=' * 70)
    print('扩界前后对比')
    print('=' * 70)
    _print_comparison(old_results, ep, front, R0_list, sel, rec_j, rec_i, old_point, old_R0)

    # --- 触碰检查（扩界后是否仍触碰） ---
    print('\n[7] 扩界后触碰检查...')
    for tag, r_opt, mname in [('联合', rec_j, 'joint'), ('独立', rec_i, 'independent')]:
        if r_opt.get('bound_touch'):
            print(f'  {tag} 扩界后仍触碰: {r_opt["bound_touch"]} — 需进一步扩大或标记')
        else:
            print(f'  {tag} 未触碰 ✓')

    return {'ep': ep, 'front': front, 'selection': sel, 'checks': checks,
            'rec_joint': rec_j, 'rec_indep': rec_i, 'R0_list': R0_list}


def _load_old_results():
    """尝试读取旧 output_pareto 目录的结果用于对比。"""
    old_dir = os.path.join(_HERE, 'output_pareto')
    old = {}
    # 尝试读旧的整数前沿表
    old_path = os.path.join(old_dir, 'Pareto工程整数最优.xlsx')
    if os.path.exists(old_path):
        try:
            old['df_int'] = pd.read_excel(old_path)
        except Exception:
            old['df_int'] = None
    # 尝试读旧的推荐方案
    old_path2 = os.path.join(old_dir, '推荐方案经济汇总_联合.xlsx')
    if os.path.exists(old_path2):
        try:
            old['rec_joint'] = pd.read_excel(old_path2)
        except Exception:
            old['rec_joint'] = None
    old_path3 = os.path.join(old_dir, '推荐方案经济汇总_独立.xlsx')
    if os.path.exists(old_path3):
        try:
            old['rec_indep'] = pd.read_excel(old_path3)
        except Exception:
            old['rec_indep'] = None
    return old


def _print_comparison(old_results, ep, front, R0_list, sel, rec_j, rec_i, old_point, old_R0):
    """打印扩界前后对比。"""
    new_R0 = sel['best']['R0']
    new_label = f'{new_R0*100:.2f}%' if new_R0 is not None else '纯经济'

    # 1. 扫描范围对比
    print(f'\n--- 扫描范围 ---')
    print(f'  旧: [88.30%, 98.19%]  新: [{ep["ind_econ_c"]["R_load"]*100:.2f}%'
          f' econ → {min(ep["ind_max_c"]["R_load"], ep["jnt_max_c"]["R_load"])*100:.2f}% max]')

    # 2. 折中点对比
    print(f'\n--- 折中点 ---')
    print(f'  旧: 94.22%  新: {new_label}')
    print(f'  距离变化: 新折中点理想距离={sel["best"]["ideal_dist"]:.4f}')

    # 3. 联合推荐方案对比
    print(f'\n--- 联合推荐方案 ---')
    dKpv_j, dKw_j, Pess_j, Eess_j = P._sum_caps(rec_j)
    print(f'  新: ΔPV={dKpv_j:.0f}kW  ΔWind={dKw_j:.0f}kW  P_ess={Pess_j:.0f}kW  E_ess={Eess_j:.0f}kWh  '
          f'C={rec_j["annual_cost"]:.0f}元  R_load={rec_j["R_load"]*100:.3f}%')

    # 4. 独立推荐方案对比
    print(f'\n--- 独立推荐方案 ---')
    dKpv_i, dKw_i, Pess_i, Eess_i = P._sum_caps(rec_i)
    print(f'  新: ΔPV={dKpv_i:.0f}kW  ΔWind={dKw_i:.0f}kW  P_ess={Pess_i:.0f}kW  E_ess={Eess_i:.0f}kWh  '
          f'C={rec_i["annual_cost"]:.0f}元  R_load={rec_i["R_load"]*100:.3f}%')

    # 5. 联合收益
    dC_new = rec_i['annual_cost'] - rec_j['annual_cost']
    eta_new = dC_new / rec_i['annual_cost'] * 100
    print(f'\n--- 联合收益 ---')
    print(f'  新: ΔC={dC_new:.0f}元  η_C={eta_new:.3f}%')
    print(f'  旧: ΔC=1,580,929元  η_C=15.34%')

    # 6. 若新折中点≠94.22%，对比两点的联合成本
    if new_R0 is not None and abs(new_R0 - old_R0) > 0.005:
        print(f'\n--- 折中点漂移分析 ---')
        # 从 front 中找最接近 94.22% 的点
        closest = min(R0_list, key=lambda r: abs(r - old_R0))
        if closest in front and 'jnt_i' in front[closest]:
            r_old = front[closest]['jnt_i']
            cost_diff = rec_j['annual_cost'] - r_old['annual_cost']
            cost_pct = cost_diff / r_old['annual_cost'] * 100
            print(f'  最近点 R0={closest*100:.2f}%: C={r_old["annual_cost"]:.0f}元')
            print(f'  新折中 C={rec_j["annual_cost"]:.0f}元  成本差={cost_diff:.0f}元 ({cost_pct:+.2f}%)')
            if abs(cost_pct) < 1.0:
                print(f'  结论: 成本差 <1%，可保留 94.22% 作为推荐')
            else:
                print(f'  结论: 成本差 ≥1%，建议采用新折中点 {new_label}')
        elif old_point is not None and 'jnt_i' in old_point:
            r_old = old_point['jnt_i']
            cost_diff = rec_j['annual_cost'] - r_old['annual_cost']
            cost_pct = cost_diff / r_old['annual_cost'] * 100
            print(f'  旧 94.22% 扩界后联合: C={r_old["annual_cost"]:.0f}元')
            print(f'  新折中 C={rec_j["annual_cost"]:.0f}元  成本差={cost_diff:.0f}元 ({cost_pct:+.2f}%)')
            if abs(cost_pct) < 1.0:
                print(f'  结论: 成本差 <1%，可保留 94.22% 作为推荐')
            else:
                print(f'  结论: 成本差 ≥1%，建议采用新折中点 {new_label}')
        else:
            print(f'  （旧 94.22% 点不在新前沿范围内，无法直接对比）')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-integers', action='store_true', help='仅跑连续，快速验证')
    args = ap.parse_args()
    if args.no_integers:
        print('仅连续模式（快速验证）')
        # 简化版：仅连续
        M.DK_MAX = NEW_DK_MAX
        M.P_MAX_IND = NEW_P_MAX_IND
        M.E_MAX_IND = NEW_E_MAX_IND
        M.P_MAX_JNT = NEW_P_MAX_JNT
        M.E_MAX_JNT = NEW_E_MAX_JNT
        data = prepare_data()
        ep = P.solve_endpoints(data)
        lo, hi = P.scan_range(ep)
        print(f'扫描范围: [{lo*100:.3f}%, {hi*100:.3f}%]')
        R0_list = list(np.linspace(lo, hi, P.R0_NPOINTS))
        front = P.build_front(data, R0_list, integers=False)
        sel = P.ideal_point_select(ep, front, R0_list)
        print(f'推荐折中: {sel["best"]["label"]}')
    else:
        run()
