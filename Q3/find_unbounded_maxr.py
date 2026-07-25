# -*- coding: utf-8 -*-
"""找到 A/B 园区不触界的 max R_load 端点，用于确定最终容量上界。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader_12m import load_12m_data, load_load_12m

data = load_12m_data()
loads = load_load_12m()

import q3_part2_model as m

# 设置极大上界，确保不触界
m.DK_MAX = 100000
m.P_MAX = 100000
m.E_MAX = 200000
m.BIG_M = 500000.0

print("=" * 60)
print("查找无约束 max R_load 端点 (DK_MAX=50000, P_MAX=20000, E_MAX=80000)")
print("=" * 60)

for park_id in ['A', 'B']:
    phi_pv = data['phi_pv'].get(park_id)
    phi_w = data['phi_w'].get(park_id)
    load = loads[park_id]

    print(f"\n--- {park_id}: max R_load (unbounded) ---")
    r = m.solve_park(park_id, phi_pv, phi_w, load, mode='max_rload', R0=None,
                     integer_cap=False, time_limit=600, verbose=True)

    caps = r['capacities']
    print(f"  R_load={r['annual']['R_load']:.4f}")
    print(f"  新增: dKpv={caps['dK_pv']:.0f}kW, dKw={caps['dK_w']:.0f}kW")
    print(f"  储能: P={caps['P_ess']:.0f}kW, E={caps['E_ess']:.0f}kWh, E/P={caps['E_ess']/max(caps['P_ess'],1e-6):.1f}h")
    print(f"  触界: {r['bound_touch']}")
    print(f"  年成本={r['annual']['total_cost']/1e4:.2f}万")

    # 计算需要的上界倍率
    orig = {'DK': 4000, 'P': 2000, 'E': 8000}
    mult_dk = caps['dK_pv'] / orig['DK'] if caps['dK_pv'] > 0 else caps['dK_w'] / orig['DK']
    mult_p = caps['P_ess'] / orig['P']
    mult_e = caps['E_ess'] / orig['E']
    print(f"  需要倍率: DK≥{mult_dk:.1f}x, P≥{mult_p:.1f}x, E≥{mult_e:.1f}x")
