# -*- coding: utf-8 -*-
"""
问题3（2）Pareto 前端构建脚本

ε-约束法：对园区 A/B/C 分别扫描 9 档 R₀（含两端）。
每档做连续容量 + 工程整数容量。
等权理想点选取折中方案，附加统一 R₀ 横向对比。

运行:
    $env:PYTHONUTF8='1'; & <math-python> Q3/q3_part2_pareto.py
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader_12m import (
    load_12m_data, load_load_12m, DAYS_PER_MONTH,
    K0_PV, K0_W, C_PV, C_W, C_P_ESS, C_E_ESS, grid_price,
)
from q3_part2_model import solve_park, solve_max_green_two_stage, DK_MAX, P_MAX, E_MAX

# 扩大容量上界（覆盖 B max green 全端点）
DK_MAX_ORIG = DK_MAX
P_MAX_ORIG = P_MAX
E_MAX_ORIG = E_MAX
import q3_part2_model as _m
_m.DK_MAX = 10000
_m.P_MAX = 8000
_m.E_MAX = 20000
_m.BIG_M = 50000.0
print(f"[bounds] 扩容: DK {DK_MAX_ORIG}→10000, P {P_MAX_ORIG}→8000, E {E_MAX_ORIG}→20000")

# 输出目录
OUT_DIR = os.path.join(_HERE, 'output_pareto_part2')
os.makedirs(OUT_DIR, exist_ok=True)

# Pareto 点数（含两端）
N_R0_TARGETS = 9
PARK_NAMES = {'A': '园区A', 'B': '园区B', 'C': '园区C'}


# =====================================================================
# 步骤 1: 求端点
# =====================================================================
def find_endpoints(park_id, phi_pv, phi_w, load, verbose=True):
    """返回 (econ_result, max_rload_result) — 均为连续容量。
    max_rload 使用两阶段求解：先 max R_load，再 min cost @ R_load_max。"""
    if verbose:
        print(f"\n  [{park_id}] 经济最优 ...")
    econ = solve_park(park_id, phi_pv, phi_w, load, mode='cost', R0=None,
                      integer_cap=False, time_limit=180, verbose=verbose)

    if verbose:
        print(f"  [{park_id}] 最大 R_load (两阶段) ...")
    rmax = solve_max_green_two_stage(park_id, phi_pv, phi_w, load,
                                     integer_cap=False, time_limit=300, verbose=verbose)

    return econ, rmax


# =====================================================================
# 步骤 2: Pareto 扫描
# =====================================================================
def scan_pareto(park_id, phi_pv, phi_w, load, R0_econ, R0_max, verbose=True):
    """对 [R0_econ, R0_max] 区间生成 N_R0_TARGETS 档 R₀，每档连续+整数求解。"""
    R0_list = np.linspace(R0_econ, R0_max, N_R0_TARGETS)

    results_cont = []
    results_int = []

    for i, R0 in enumerate(R0_list):
        if verbose:
            print(f"  [{park_id}] R0={R0:.4f} ({i+1}/{N_R0_TARGETS}) ...")

        # 连续容量
        r_cont = solve_park(park_id, phi_pv, phi_w, load, mode='cost', R0=float(R0),
                            integer_cap=False, time_limit=180)
        results_cont.append(r_cont)

        # 整数容量
        r_int = solve_park(park_id, phi_pv, phi_w, load, mode='cost', R0=float(R0),
                           integer_cap=True, time_limit=180)
        results_int.append(r_int)

    return R0_list, results_cont, results_int


# =====================================================================
# 步骤 3: 理想点折中选取
# =====================================================================
def ideal_point_select(results):
    """从 Pareto 前沿结果中选取等权归一化理想点距离最小的点。

    理想点 = (min cost, max R_load)
    对每个点计算归一化欧氏距离，返回最佳点索引。
    """
    costs = np.array([r['annual']['total_cost'] for r in results])
    rloads = np.array([r['annual']['R_load'] for r in results])

    c_min, c_max = costs.min(), costs.max()
    r_min, r_max = rloads.min(), rloads.max()

    if c_max - c_min < 1e-6:
        cost_norm = np.zeros_like(costs)
    else:
        cost_norm = (costs - c_min) / (c_max - c_min)

    if r_max - r_min < 1e-6:
        r_norm = np.zeros_like(rloads)
    else:
        r_norm = (r_max - rloads) / (r_max - r_min)  # 反向：越大越好

    dist = np.sqrt(cost_norm**2 + r_norm**2)
    best_idx = int(np.argmin(dist))

    return best_idx, {
        'ideal_cost': c_min,
        'ideal_rload': r_max,
        'distances': dist.tolist(),
        'cost_norm': cost_norm.tolist(),
        'rload_norm': r_norm.tolist(),
    }


# =====================================================================
# 边际成本计算
# =====================================================================
def marginal_cost_table(results):
    """计算相邻 R₀ 档之间的边际成本（元/百分点 R_load 提升）。"""
    rows = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        dc = curr['annual']['total_cost'] - prev['annual']['total_cost']
        dr = (curr['annual']['R_load'] - prev['annual']['R_load']) * 100  # 百分点
        mc = dc / dr if abs(dr) > 1e-8 else float('inf')
        rows.append({
            'from_R0': prev['R0'],
            'to_R0': curr['R0'],
            'from_R_load': prev['annual']['R_load'],
            'to_R_load': curr['annual']['R_load'],
            'delta_cost': dc,
            'delta_R_load_pct': dr,
            'marginal_cost_per_pct': mc,
        })
    return rows


# =====================================================================
# 统一 R₀ 横向对比
# =====================================================================
def unified_R0_comparison(all_results, common_R0, phi_data, loads):
    """在统一 R₀ 下求解三个园区（连续容量），返回对比结果。"""
    comparison = {}
    for park_id in ['A', 'B', 'C']:
        phi_pv = phi_data['phi_pv'].get(park_id)
        phi_w = phi_data['phi_w'].get(park_id)
        load = loads[park_id]
        r = solve_park(park_id, phi_pv, phi_w, load, mode='cost', R0=common_R0,
                       integer_cap=False, time_limit=180, verbose=True)
        comparison[park_id] = r
    return comparison


# =====================================================================
# 输出格式化
# =====================================================================
def build_summary_table(park_results, label):
    """构建单个园区的 Pareto 汇总 DataFrame。"""
    rows = []
    for r in park_results:
        rows.append({
            'R0': r['R0'],
            '模式': label,
            'ΔKpv(kW)': r['capacities']['dK_pv'],
            'ΔKw(kW)': r['capacities']['dK_w'],
            'P_ess(kW)': r['capacities']['P_ess'],
            'E_ess(kWh)': r['capacities']['E_ess'],
            'E/P(h)': r['capacities']['E_ess'] / max(r['capacities']['P_ess'], 1e-6),
            '年运行成本(万元)': r['annual']['running_cost'] / 10000,
            '年投资-风光(万元)': r['annual']['inv_gen'] / 10000,
            '年投资-储能(万元)': r['annual']['inv_ess'] / 10000,
            '年总成本(万元)': r['annual']['total_cost'] / 10000,
            'R_load': r['annual']['R_load'],
            'R_re': r['annual']['R_re'],
            '年购电量(MWh)': r['annual']['grid_total'] / 1000,
            '年弃电量(MWh)': r['annual']['curt_total'] / 1000,
            '峰段购电(MWh)': r['annual']['peak_grid'] / 1000,
            '谷段购电(MWh)': r['annual']['valley_grid'] / 1000,
            '求解时间(s)': r['solve_time'],
            '截断': ','.join(r['bound_touch']) if r['bound_touch'] else '',
        })
    return pd.DataFrame(rows)


def build_marginal_table(all_marginals):
    """合并所有园区边际成本表。"""
    rows = []
    for park_id, mc_list in all_marginals.items():
        for mc in mc_list:
            rows.append({
                '园区': park_id,
                'from_R0': mc['from_R0'],
                'to_R0': mc['to_R0'],
                'Δ成本(万元)': mc['delta_cost'] / 10000,
                'ΔR_load(百分点)': mc['delta_R_load_pct'],
                '边际成本(元/百分点)': mc['marginal_cost_per_pct'],
            })
    return pd.DataFrame(rows)


# =====================================================================
# 主流程
# =====================================================================
def main():
    print("=" * 70)
    print("Q3(2) Pareto 前端构建")
    print("=" * 70)

    # ---- 加载数据 ----
    print("\n[1/5] 加载数据 ...")
    phi_data = load_12m_data()
    loads = load_load_12m()

    all_endpoints = {}     # {park: (econ, rmax)}
    all_pareto_cont = {}   # {park: {R0_list, results}}
    all_pareto_int = {}    # {park: {R0_list, results}}
    all_compromise = {}    # {park: {idx, info, result}}
    all_marginals = {}     # {park: marginal_table}

    # ---- 步骤 1: 端点 ----
    print("\n[2/5] 求 Pareto 端点 ...")
    for park_id in ['A', 'B', 'C']:
        print(f"\n--- {PARK_NAMES[park_id]} ({park_id}) ---")
        phi_pv = phi_data['phi_pv'].get(park_id)
        phi_w = phi_data['phi_w'].get(park_id)
        load = loads[park_id]

        econ, rmax = find_endpoints(park_id, phi_pv, phi_w, load)
        all_endpoints[park_id] = (econ, rmax)

        R0_econ = econ['annual']['R_load']
        R0_max = rmax['annual']['R_load']
        print(f"  经济最优: R_load={R0_econ:.4f}, cost={econ['annual']['total_cost']/1e4:.2f}万")
        print(f"  最大R_load: {R0_max:.4f}, cost={rmax['annual']['total_cost']/1e4:.2f}万")
        print(f"  扫描范围: [{R0_econ:.4f}, {R0_max:.4f}]")

    # ---- 步骤 2: Pareto 扫描 ----
    print(f"\n[3/5] Pareto 扫描 ({N_R0_TARGETS} 档 R₀) ...")
    t0 = time.time()

    for park_id in ['A', 'B', 'C']:
        print(f"\n--- {PARK_NAMES[park_id]} ({park_id}) ---")
        phi_pv = phi_data['phi_pv'].get(park_id)
        phi_w = phi_data['phi_w'].get(park_id)
        load = loads[park_id]
        econ, rmax = all_endpoints[park_id]

        R0_list, res_cont, res_int = scan_pareto(
            park_id, phi_pv, phi_w, load,
            econ['annual']['R_load'], rmax['annual']['R_load']
        )

        all_pareto_cont[park_id] = {'R0_list': R0_list, 'results': res_cont}
        all_pareto_int[park_id] = {'R0_list': R0_list, 'results': res_int}

        # 理想点选取（用整数结果作主结果）
        best_idx, ideal_info = ideal_point_select(res_int)
        all_compromise[park_id] = {
            'idx': best_idx,
            'info': ideal_info,
            'result': res_int[best_idx],
        }

        # 边际成本
        all_marginals[park_id] = marginal_cost_table(res_int)

        print(f"  折中点: R0*={R0_list[best_idx]:.4f}, "
              f"R_load={res_int[best_idx]['annual']['R_load']:.4f}, "
              f"cost={res_int[best_idx]['annual']['total_cost']/1e4:.2f}万")

    elapsed = time.time() - t0
    print(f"\n  扫描完成，总耗时 {elapsed:.1f}s")

    # ---- 步骤 4: 统一 R₀ 横向对比 ----
    print("\n[4/5] 统一 R₀ 横向对比 ...")

    # 取三个园区 R_load_econ 的最大值作为最低可行统一 R₀
    R0_econ_vals = [all_endpoints[p][0]['annual']['R_load'] for p in ['A', 'B', 'C']]
    common_R0_min = max(R0_econ_vals)  # 所有园区都能达到的最低水平
    # 取三个园区 R_load_max 的最小值作为最高可达统一 R₀
    R0_max_vals = [all_endpoints[p][1]['annual']['R_load'] for p in ['A', 'B', 'C']]
    common_R0_max = min(R0_max_vals)

    print(f"  统一可行范围: [{common_R0_min:.4f}, {common_R0_max:.4f}]")

    n_unified = 5
    unified_R0s = np.linspace(common_R0_min, common_R0_max, n_unified)
    unified_results = {}

    for R0u in unified_R0s:
        print(f"\n  统一 R₀={R0u:.4f} ...")
        unified_results[R0u] = unified_R0_comparison(
            all_pareto_cont, float(R0u), phi_data, loads
        )

    # ---- 步骤 5: 输出结果 ----
    print("\n[5/5] 输出结果 ...")

    # 主要 Excel 输出
    with pd.ExcelWriter(os.path.join(OUT_DIR, 'Pareto结果汇总.xlsx'), engine='openpyxl') as writer:
        # 每个园区连续 + 整数表
        for park_id in ['A', 'B', 'C']:
            df_cont = build_summary_table(all_pareto_cont[park_id]['results'], '连续')
            df_int = build_summary_table(all_pareto_int[park_id]['results'], '整数')
            df_all = pd.concat([df_cont, df_int], ignore_index=True)
            df_all.to_excel(writer, sheet_name=f'{park_id}_全部结果', index=False)

        # 工程整数结果汇总
        df_int_summary = pd.DataFrame()
        for park_id in ['A', 'B', 'C']:
            df = build_summary_table(all_pareto_int[park_id]['results'], '整数')
            df.insert(0, '园区', park_id)
            df_int_summary = pd.concat([df_int_summary, df], ignore_index=True)
        df_int_summary.to_excel(writer, sheet_name='整数结果汇总', index=False)

        # 折中方案
        compromise_rows = []
        for park_id in ['A', 'B', 'C']:
            r = all_compromise[park_id]['result']
            compromise_rows.append({
                '园区': park_id,
                'R0': r['R0'],
                'ΔKpv(kW)': r['capacities']['dK_pv'],
                'ΔKw(kW)': r['capacities']['dK_w'],
                'P_ess(kW)': r['capacities']['P_ess'],
                'E_ess(kWh)': r['capacities']['E_ess'],
                'E/P(h)': r['capacities']['E_ess'] / max(r['capacities']['P_ess'], 1e-6),
                '年总成本(万元)': r['annual']['total_cost'] / 10000,
                'R_load': r['annual']['R_load'],
                'R_re': r['annual']['R_re'],
                '折中距离': all_compromise[park_id]['info']['distances'][all_compromise[park_id]['idx']],
            })
        pd.DataFrame(compromise_rows).to_excel(writer, sheet_name='折中方案', index=False)

        # 边际成本
        df_mc = build_marginal_table(all_marginals)
        df_mc.to_excel(writer, sheet_name='边际成本', index=False)

        # 统一 R₀ 对比
        unified_rows = []
        for R0u, comp in unified_results.items():
            for park_id in ['A', 'B', 'C']:
                r = comp[park_id]
                unified_rows.append({
                    '统一R₀': R0u,
                    '园区': park_id,
                    '年总成本(万元)': r['annual']['total_cost'] / 10000,
                    'R_load': r['annual']['R_load'],
                    'R_re': r['annual']['R_re'],
                    'ΔKpv(kW)': r['capacities']['dK_pv'],
                    'ΔKw(kW)': r['capacities']['dK_w'],
                    'P_ess(kW)': r['capacities']['P_ess'],
                    'E_ess(kWh)': r['capacities']['E_ess'],
                    '年购电(MWh)': r['annual']['grid_total'] / 1000,
                    '年弃电(MWh)': r['annual']['curt_total'] / 1000,
                })
        pd.DataFrame(unified_rows).to_excel(writer, sheet_name='统一R0对比', index=False)

    # 详细结果 JSON
    detailed = {
        'endpoints': {},
        'compromise': {},
    }
    for park_id in ['A', 'B', 'C']:
        econ, rmax = all_endpoints[park_id]
        detailed['endpoints'][park_id] = {
            'econ': {k: v for k, v in econ.items() if k != 'monthly'},
            'max_rload': {k: v for k, v in rmax.items() if k != 'monthly'},
        }
        comp = all_compromise[park_id]
        detailed['compromise'][park_id] = {
            'idx': comp['idx'],
            'info': comp['info'],
            'result': {k: v for k, v in comp['result'].items() if k != 'monthly'},
        }

    with open(os.path.join(OUT_DIR, 'detailed_results.json'), 'w', encoding='utf-8') as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2, default=str)

    # ---- 控制台输出 ----
    print("\n" + "=" * 70)
    print("结果摘要")
    print("=" * 70)

    for park_id in ['A', 'B', 'C']:
        econ = all_endpoints[park_id][0]
        comp = all_compromise[park_id]['result']
        rmax = all_endpoints[park_id][1]

        print(f"\n## {PARK_NAMES[park_id]} ({park_id})")
        print(f"  {'':>12} {'经济最优':>15} {'折中方案':>15} {'最大R_load':>15}")
        print(f"  {'ΔKpv(kW)':>12} {econ['capacities']['dK_pv']:15.0f} "
              f"{comp['capacities']['dK_pv']:15.0f} {rmax['capacities']['dK_pv']:15.0f}")
        print(f"  {'ΔKw(kW)':>12} {econ['capacities']['dK_w']:15.0f} "
              f"{comp['capacities']['dK_w']:15.0f} {rmax['capacities']['dK_w']:15.0f}")
        print(f"  {'P_ess(kW)':>12} {econ['capacities']['P_ess']:15.0f} "
              f"{comp['capacities']['P_ess']:15.0f} {rmax['capacities']['P_ess']:15.0f}")
        print(f"  {'E_ess(kWh)':>12} {econ['capacities']['E_ess']:15.0f} "
              f"{comp['capacities']['E_ess']:15.0f} {rmax['capacities']['E_ess']:15.0f}")
        print(f"  {'年成本(万元)':>12} {econ['annual']['total_cost']/1e4:15.2f} "
              f"{comp['annual']['total_cost']/1e4:15.2f} {rmax['annual']['total_cost']/1e4:15.2f}")
        print(f"  {'R_load':>12} {econ['annual']['R_load']:15.4f} "
              f"{comp['annual']['R_load']:15.4f} {rmax['annual']['R_load']:15.4f}")
        print(f"  {'R_re':>12} {econ['annual']['R_re']:15.4f} "
              f"{comp['annual']['R_re']:15.4f} {rmax['annual']['R_re']:15.4f}")

    print(f"\n结果已保存至: {OUT_DIR}")
    print("完成 ✓")


if __name__ == '__main__':
    main()
