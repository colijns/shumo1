# -*- coding: utf-8 -*-
"""Q3(2) Bound-touch re-expansion check. 对触界方案扩大 1.5× 容量上界重算。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader_12m import load_12m_data, load_load_12m

data = load_12m_data()
loads = load_load_12m()

# Monkey-patch capacity bounds BEFORE importing model
import q3_part2_model as m
ORIG = {'DK': m.DK_MAX, 'P': m.P_MAX, 'E': m.E_MAX}
m.DK_MAX = 6000
m.P_MAX = 3000
m.E_MAX = 12000

# Also patch BIG_M since larger capacities need larger M
m.BIG_M = 30000.0

print("=" * 60)
print("Bound-touch re-expansion: 1.5x")
print(f"DK_MAX: {ORIG['DK']} -> {m.DK_MAX}")
print(f"P_MAX:  {ORIG['P']} -> {m.P_MAX}")
print(f"E_MAX:  {ORIG['E']} -> {m.E_MAX}")
print("=" * 60)

for park_id in ['A', 'B', 'C']:
    phi_pv = data['phi_pv'].get(park_id)
    phi_w = data['phi_w'].get(park_id)
    load = loads[park_id]

    print(f"\n--- {park_id}: max R_load ---")
    r = m.solve_park(park_id, phi_pv, phi_w, load, mode='max_rload', R0=None,
                     integer_cap=False, time_limit=300, verbose=True)

    caps = r['capacities']
    print(f"  R_load={r['annual']['R_load']:.4f}")
    print(f"  总装机: PV={caps['K_pv_total']:.0f}kW, W={caps['K_w_total']:.0f}kW")
    print(f"  新增: ΔKpv={caps['dK_pv']:.0f}kW, ΔKw={caps['dK_w']:.0f}kW")
    print(f"  储能: P={caps['P_ess']:.0f}kW, E={caps['E_ess']:.0f}kWh, E/P={caps['E_ess']/max(caps['P_ess'],1e-6):.1f}h")
    print(f"  年成本={r['annual']['total_cost']/1e4:.2f}万")
    print(f"  R_re={r['annual']['R_re']:.4f}")
    print(f"  Bound touch: {r['bound_touch']}")
    print(f"  Errors: {len(r['errors'])}")
