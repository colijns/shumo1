# -*- coding: utf-8 -*-
"""问题2（3）联合运营相对独立运营的经济收益：三层对比（ADR 0004）。

三层结果均使用不含弃电惩罚的年实际成本：
  1. 最终方案比较：Q1 独立工程最优 (95/190) vs 联合 Pareto 推荐 (175/890)；
  2. 纯经济比较：独立纯经济最优 (95/190) vs 联合纯经济最优 (0/0)；
  3. 相同 97% 绿色标准：各园区独立 ≥97% vs 联合整体 ≥97% (175/890)。

联合运营年收益定义为  B = C_ind^ann − C_joint^ann，
收益率              r = B / C_ind^ann × 100%。

口径说明：
  - 独立侧：三园区各自 MILP 求最优运行策略，汇总（购电/弃电/成本/投资相加，
    消纳率按总风光利用/总风光发电加权）。
  - 独立纯经济：run_engineering_storage(P_max=200, E_max=600)（Q1 口径，复现 95/190）。
  - 独立 ≥97%：run_engineering_storage(P_max=500, E_max=1500, min_accom_rate=0.97)
    （大上界保证不 binding；5kW/10kWh 粒度）。
  - 联合侧：复用 q2_pareto 的 Pareto 前沿 + 理想点选型。

运行：
    $env:PYTHONUTF8='1'; & <math-python> Q2/q2_3_compare.py
    & <math-python> -m pytest Q2/test_q2_3_compare.py -v
"""

import os
import sys

import numpy as np
import pandas as pd

# ----- 跨目录复用 Q1 / Q2 / templates -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_Q2_DIR = _HERE
_TEMPLATE_DIR = os.path.abspath(os.path.join(_HERE, '..', 'templates'))
for _d in (_Q1_DIR, _Q2_DIR, _TEMPLATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import load_load_data, load_solar_wind_data  # noqa: E402
from milp_model import run_engineering_storage  # noqa: E402
from q2_main import build_joint_profile  # noqa: E402
from q2_pareto import (  # noqa: E402
    build_pareto_front, ideal_point_selection, solve_continuous_at,
    P_STEP, E_STEP,
)
from common.io_utils import export_result  # noqa: E402

PARKS = ['A', 'B', 'C']

# Q1 单园区工程上界（纯经济最优远小于此，不 binding）
P_MAX_Q1, E_MAX_Q1 = 200, 600
# 独立 ≥97% 上界：单园区无互济，达 97% 需大储能；取足够大不 binding
P_MAX_97, E_MAX_97 = 500, 1500
R0_97 = 0.97


# =====================================================================
# 独立侧求解
# =====================================================================

def solve_independent_pure_economic(load_data, G_pv, G_w):
    """各园区独立纯经济工程最优（无消纳约束），复现 Q1 的 95/190。"""
    return {p: run_engineering_storage(load_data[p], G_pv[p], G_w[p],
                                       P_step=P_STEP, E_step=E_STEP,
                                       P_max=P_MAX_Q1, E_max=E_MAX_Q1)
            for p in PARKS}


def solve_independent_at_97(load_data, G_pv, G_w):
    """各园区独立 ≥97% 消纳率工程最优。"""
    return {p: run_engineering_storage(load_data[p], G_pv[p], G_w[p],
                                       P_step=P_STEP, E_step=E_STEP,
                                       P_max=P_MAX_97, E_max=E_MAX_97,
                                       min_accom_rate=R0_97)
            for p in PARKS}


def aggregate_independent(results):
    """汇总三园区独立运营指标为单 dict。"""
    agg = {
        'P_total': sum(r['P_ess'] for r in results.values()),
        'E_total': sum(r['E_ess'] for r in results.values()),
        'daily_grid': sum(r['grid_total'] for r in results.values()),
        'daily_curt': sum(r['curt_total'] for r in results.values()),
        'daily_cost': sum(r['daily_cost'] for r in results.values()),
        'annual_operating': sum(r['daily_cost'] * 365 for r in results.values()),
        'inv_annual': sum(r['inv_annual'] for r in results.values()),
        'annual_cost': sum(r['annual_cost'] for r in results.values()),
        'load_total': sum(r['load_total'] for r in results.values()),
    }
    agg['annual_operating'] = float(agg['annual_operating'])
    agg['unit_annual_cost'] = agg['annual_cost'] / (365 * agg['load_total'])
    total_re_gen = sum(r['pv_gen'] + r['w_gen'] for r in results.values())
    total_re_use = sum(r['pv_use'] + r['w_use'] for r in results.values())
    agg['re_ratio'] = total_re_use / total_re_gen if total_re_gen > 0 else 0.0
    return agg


# =====================================================================
# 联合侧求解（复用 q2_pareto）
# =====================================================================

def solve_joint(L_J, G_pv_J, G_w_J):
    """返回 (r_joint_97, r_joint_pure, selection)。

    r_joint_97   : 联合 ≥97% 工程整数方案（Pareto 推荐 175/890）
    r_joint_pure : 联合纯经济最优（0/0，连续）
    selection    : 理想点选型结果（含 candidates/best/recommended_R0）
    """
    front = build_pareto_front(L_J, G_pv_J, G_w_J, verbose=False)
    sel = ideal_point_selection(front)
    r_joint_97 = sel['recommended_int']
    r_joint_pure = front['baseline_cont']
    return r_joint_97, r_joint_pure, sel


def aggregate_joint(r):
    """联合单方案指标 dict（联合为单一 MILP，直接取字段）。"""
    return {
        'P_total': r['P_ess'],
        'E_total': r['E_ess'],
        'daily_grid': r['grid_total'],
        'daily_curt': r['curt_total'],
        'daily_cost': r['daily_cost'],
        'annual_operating': r['daily_cost'] * 365,
        'inv_annual': r['inv_annual'],
        'annual_cost': r['annual_cost'],
        'load_total': r['load_total'],
        'unit_annual_cost': r['unit_annual_cost'],
        're_ratio': r['re_ratio'],
    }


# =====================================================================
# 对比表构造
# =====================================================================

def _row(metric, ind, joint, fmt=None):
    """一行：指标 | 独立 | 联合 | 变化。"""
    delta = joint - ind
    return {
        '指标': metric,
        '独立运营': round(ind, 6) if fmt is None else fmt(ind),
        '联合运营': round(joint, 6) if fmt is None else fmt(joint),
        '联合后变化': round(delta, 6) if fmt is None else fmt(delta),
    }


def build_comparison_table(ind_agg, jnt_agg):
    """构造一张三层对比表（指标行 = doc §2.3 格式）。"""
    f2 = lambda x: round(float(x), 3)
    f0 = lambda x: round(float(x), 2)
    f4 = lambda x: round(float(x), 4)
    rows = [
        _row('储能总功率/kW', ind_agg['P_total'], jnt_agg['P_total'], f2),
        _row('储能总容量/kWh', ind_agg['E_total'], jnt_agg['E_total'], f2),
        _row('日电网购电量/kWh', ind_agg['daily_grid'], jnt_agg['daily_grid'], f2),
        _row('日弃电量/kWh', ind_agg['daily_curt'], jnt_agg['daily_curt'], f2),
        _row('日运行成本/元', ind_agg['daily_cost'], jnt_agg['daily_cost'], f2),
        _row('年运行成本/元', ind_agg['annual_operating'], jnt_agg['annual_operating'], f0),
        _row('储能年化投资/元', ind_agg['inv_annual'], jnt_agg['inv_annual'], f0),
        _row('年实际成本/元', ind_agg['annual_cost'], jnt_agg['annual_cost'], f0),
        _row('单位负荷电量年实际成本/元·kWh⁻¹',
             ind_agg['unit_annual_cost'], jnt_agg['unit_annual_cost'], f4),
        _row('风光消纳率', ind_agg['re_ratio'], jnt_agg['re_ratio'], f4),
    ]
    return pd.DataFrame(rows)


def build_independent_97_detail(ind_97_results):
    """§2.3.4 各园区独立 ≥97% 明细：园区/P/E/实际消纳率。"""
    rows = []
    for p in PARKS:
        r = ind_97_results[p]
        rows.append({
            '园区': p,
            '储能功率/kW': int(round(r['P_ess'])),
            '储能容量/kWh': int(round(r['E_ess'])),
            '实际消纳率': round(r['re_ratio'], 6),
        })
    # 合计行
    rows.append({
        '园区': '合计',
        '储能功率/kW': int(round(sum(r['P_ess'] for r in ind_97_results.values()))),
        '储能容量/kWh': int(round(sum(r['E_ess'] for r in ind_97_results.values()))),
        '实际消纳率': '',
    })
    return pd.DataFrame(rows)


def benefit(ind_agg, jnt_agg):
    """收益 B 与收益率 r。"""
    B = ind_agg['annual_cost'] - jnt_agg['annual_cost']
    r = B / ind_agg['annual_cost'] * 100 if ind_agg['annual_cost'] > 0 else 0.0
    return B, r


# =====================================================================
# 与 docs/问题2pareto.md 预测值对比校验
# =====================================================================

# doc §2.3.2/§2.3.3/§2.3.4 的预测值（用户已声明"预测的，不是真的"，仅作对照）
DOC_PREDICT = {
    'final': {  # §2.3.2 独立工程最优 vs 联合97%
        'ind': {'P': 95, 'E': 190, 'daily_grid': 9719.693, 'daily_curt': 2659.687,
                'daily_cost': 15985.836, 'annual_operating': 5834830.02,
                'inv_annual': 41800.00, 'annual_cost': 5876630.02,
                'unit_annual_cost': 0.6884, 're_ratio': None},
        'jnt': {'P': 175, 'E': 890, 'daily_grid': 7589.870, 'daily_curt': 487.701,
                'daily_cost': 14797.536, 'annual_operating': 5401100.76,
                'inv_annual': 174200.00, 'annual_cost': 5575300.76,
                'unit_annual_cost': 0.6531, 're_ratio': 0.97019},
        'B': 301329.25, 'r': 5.13,
    },
    'economic': {  # §2.3.3 纯经济
        'ind': {'P': 95, 'E': 190, 'daily_grid': 9719.693, 'daily_curt': 2659.687,
                'annual_operating': 5834830.02, 'inv_annual': 41800.00,
                'annual_cost': 5876630.02},
        'jnt': {'P': 0, 'E': 0, 'daily_grid': 8266.270, 'daily_curt': 1237.175,
                'annual_operating': 5514296.08, 'inv_annual': 0,
                'annual_cost': 5514296.08},
        'B': 362333.94, 'r': 6.17,
    },
    '97': {  # §2.3.4 同97%
        'ind': {'P': 515, 'E': 2790, 'daily_grid': 7772.407, 'daily_curt': 487.831,
                'daily_cost': 14991.760, 'annual_operating': None,
                'inv_annual': 543400.00, 'annual_cost': 6015392.33},
        'jnt': {'P': 175, 'E': 890, 'daily_grid': 7589.870, 'daily_curt': 487.701,
                'daily_cost': 14797.536, 'annual_operating': 5401100.76,
                'inv_annual': 174200.00, 'annual_cost': 5575300.76},
        'B': 440091.57, 'r': 7.32,
        'ind_detail': {'A': (160, 990, 0.97046), 'B': (165, 750, 0.97000),
                       'C': (190, 1050, 0.97017)},
    },
}


def _cmp(label, actual, predict, tol):
    if predict is None:
        return None
    ok = abs(actual - predict) <= tol
    flag = '✓' if ok else '✗'
    print(f"    {flag} {label}: 实测={actual:.4f}  预测={predict:.4f}  "
          f"差={actual - predict:+.4f}")
    return ok


def verify_against_doc(final_ind, final_jnt, econ_ind, econ_jnt,
                       q97_ind, q97_jnt, q97_detail_rows):
    """与 doc 预测值逐项对照，返回通过数/总数。"""
    print("\n  [校验] 与 docs/问题2pareto.md 预测值对照（tol: 成本1元、率1e-3、配置整数精确）")
    checks = []

    def cmp_group(name, ind, jnt, pred, B, r):
        print(f"\n  · {name}")
        T_COST, T_RATE, T_CFG = 1.0, 1e-3, 0.0
        m = {'P': ('P_total', T_CFG), 'E': ('E_total', T_CFG),
             'daily_grid': ('daily_grid', T_COST),
             'daily_curt': ('daily_curt', T_COST), 'daily_cost': ('daily_cost', T_COST),
             'annual_operating': ('annual_operating', T_COST),
             'inv_annual': ('inv_annual', T_COST), 'annual_cost': ('annual_cost', T_COST),
             'unit_annual_cost': ('unit_annual_cost', 1e-3),
             're_ratio': ('re_ratio', T_RATE)}
        for k, (key, tol) in m.items():
            if pred['ind'].get(k) is not None:
                checks.append(_cmp(f"独立.{k}", ind[key], pred['ind'][k], tol))
            if pred['jnt'].get(k) is not None:
                checks.append(_cmp(f"联合.{k}", jnt[key], pred['jnt'][k], tol))
        B_act, r_act = benefit(ind, jnt)
        checks.append(_cmp(f"收益B", B_act, B, T_COST))
        checks.append(_cmp(f"收益率r%", r_act, r, 0.05))

    cmp_group('§2.3.2 最终方案', final_ind, final_jnt, DOC_PREDICT['final'],
              DOC_PREDICT['final']['B'], DOC_PREDICT['final']['r'])
    cmp_group('§2.3.3 纯经济', econ_ind, econ_jnt, DOC_PREDICT['economic'],
              DOC_PREDICT['economic']['B'], DOC_PREDICT['economic']['r'])
    cmp_group('§2.3.4 同97%', q97_ind, q97_jnt, DOC_PREDICT['97'],
              DOC_PREDICT['97']['B'], DOC_PREDICT['97']['r'])

    # §2.3.4 各园区明细
    print("\n  · §2.3.4 各园区独立≥97% 明细")
    pred_det = DOC_PREDICT['97']['ind_detail']
    for row in q97_detail_rows:
        if row['园区'] == '合计':
            continue
        p = row['园区']
        Pp, Ep, Rp = pred_det[p]
        checks.append(_cmp(f"  园区{p}.P", row['储能功率/kW'], Pp, 0.0))
        checks.append(_cmp(f"  园区{p}.E", row['储能容量/kWh'], Ep, 0.0))
        checks.append(_cmp(f"  园区{p}.消纳率", row['实际消纳率'], Rp, 1e-3))

    passed = sum(1 for c in checks if c)
    total = sum(1 for c in checks if c is not None)
    failed = total - passed
    print(f"\n  [校验结果] {passed}/{total} 通过，{failed} 项偏差超容差")
    return passed, total


# =====================================================================
# 主流程
# =====================================================================

def run_q2_3_compare(output_dir=None):
    """三层对比主流程。"""
    print("=" * 70)
    print("问题2（3）联合运营相对独立运营的经济收益 —— 三层对比")
    print("=" * 70)

    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)

    # --- 独立侧 ---
    print("\n[1] 独立纯经济工程最优（Q1 口径，无消纳约束）...")
    ind_pure_res = solve_independent_pure_economic(load_data, G_pv_data, G_w_data)
    for p in PARKS:
        r = ind_pure_res[p]
        print(f"    园区{p}: P={r['P_ess']:.0f}, E={r['E_ess']:.0f}, "
              f"年实际={r['annual_cost']:.2f}, 消纳率={r['re_ratio']:.6f}")

    print("\n[2] 独立各园区 ≥97% 消纳率工程最优...")
    ind_97_res = solve_independent_at_97(load_data, G_pv_data, G_w_data)
    for p in PARKS:
        r = ind_97_res[p]
        print(f"    园区{p}: P={r['P_ess']:.0f}, E={r['E_ess']:.0f}, "
              f"年实际={r['annual_cost']:.2f}, 消纳率={r['re_ratio']:.6f}")
        # 上界不 binding 检查
        assert r['P_ess'] < P_MAX_97 - 1e-6, f"园区{p} P 触上界 {P_MAX_97}"
        assert r['E_ess'] < E_MAX_97 - 1e-6, f"园区{p} E 触上界 {E_MAX_97}"

    # --- 联合侧 ---
    print("\n[3] 联合 Pareto 前沿 + 理想点选型...")
    r_jnt_97, r_jnt_pure, sel = solve_joint(L_J, G_pv_J, G_w_J)
    print(f"    联合97%推荐: R0*={sel['recommended_R0']*100:.0f}%, "
          f"P={r_jnt_97['P_ess']:.0f}, E={r_jnt_97['E_ess']:.0f}, "
          f"年实际={r_jnt_97['annual_cost']:.2f}, 消纳率={r_jnt_97['re_ratio']:.6f}")
    print(f"    联合纯经济: P={r_jnt_pure['P_ess']:.3f}, E={r_jnt_pure['E_ess']:.3f}, "
          f"年实际={r_jnt_pure['annual_cost']:.2f}")

    # --- 汇总 ---
    ind_pure_agg = aggregate_independent(ind_pure_res)
    ind_97_agg = aggregate_independent(ind_97_res)
    jnt_97_agg = aggregate_joint(r_jnt_97)
    jnt_pure_agg = aggregate_joint(r_jnt_pure)

    # --- 三张对比表 ---
    tbl_final = build_comparison_table(ind_pure_agg, jnt_97_agg)
    tbl_econ = build_comparison_table(ind_pure_agg, jnt_pure_agg)
    tbl_97 = build_comparison_table(ind_97_agg, jnt_97_agg)
    det_97 = build_independent_97_detail(ind_97_res)

    print("\n" + "=" * 70)
    print("§2.3.2 最终方案比较（独立工程最优 vs 联合97%推荐）")
    print("=" * 70)
    print(tbl_final.to_string(index=False))
    Bf, rf = benefit(ind_pure_agg, jnt_97_agg)
    print(f"\n  收益 B_final = {Bf:,.2f} 元/年,  收益率 r_final ≈ {rf:.2f}%")

    print("\n" + "=" * 70)
    print("§2.3.3 纯经济比较（独立纯经济 vs 联合纯经济）")
    print("=" * 70)
    print(tbl_econ.to_string(index=False))
    Be, re = benefit(ind_pure_agg, jnt_pure_agg)
    print(f"\n  收益 B_economic = {Be:,.2f} 元/年,  收益率 r_economic ≈ {re:.2f}%")

    print("\n" + "=" * 70)
    print("§2.3.4 相同97%绿色标准比较（独立各园区≥97% vs 联合整体≥97%）")
    print("=" * 70)
    print(det_97.to_string(index=False))
    print()
    print(tbl_97.to_string(index=False))
    B9, r9 = benefit(ind_97_agg, jnt_97_agg)
    print(f"\n  收益 B_97% = {B9:,.2f} 元/年,  收益率 r_97% ≈ {r9:.2f}%")

    # --- 与 doc 预测值对照 ---
    det_rows = det_97.to_dict('records')
    verify_against_doc(ind_pure_agg, jnt_97_agg, ind_pure_agg, jnt_pure_agg,
                       ind_97_agg, jnt_97_agg, det_rows)

    # --- 导出 ---
    if output_dir is None:
        output_dir = os.path.join(_HERE, 'output_q2_3')
    os.makedirs(output_dir, exist_ok=True)
    export_result(tbl_final, os.path.join(output_dir, '最终方案对比.xlsx'))
    export_result(tbl_econ, os.path.join(output_dir, '纯经济对比.xlsx'))
    export_result(tbl_97, os.path.join(output_dir, '同97%标准对比.xlsx'))
    export_result(det_97, os.path.join(output_dir, '独立各园区97%明细.xlsx'))
    print(f"\n  [导出] 三张对比表 + 明细表 -> {output_dir}")

    return {
        'tbl_final': tbl_final, 'tbl_econ': tbl_econ, 'tbl_97': tbl_97,
        'det_97': det_97,
        'benefit_final': (Bf, rf), 'benefit_economic': (Be, re),
        'benefit_97': (B9, r9),
        'ind_pure_res': ind_pure_res, 'ind_97_res': ind_97_res,
        'r_jnt_97': r_jnt_97, 'r_jnt_pure': r_jnt_pure, 'selection': sel,
    }


if __name__ == '__main__':
    run_q2_3_compare()
