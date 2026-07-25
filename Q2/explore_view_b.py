# -*- coding: utf-8 -*-
"""
View B 探索（一次性尝试，不入主模型）：

口径：储能充电消耗的是"本要弃掉的风光余电"，机会成本 = 0，充电免费。
      风光直供负荷仍按 0.4/0.5 计；电网购电 1.0；弃电仍免费（curt_penalty=0）。
      surplus_only_charge=True 保证充电只能用 max(G_pv+G_w-L, 0)，即"优先用弃电"。

对比：
  View A（主模型，ADR 0002）：充电按 0.4/0.5 计  -> 纯经济最优 0/0
  View B（本探索）           ：充电计 0          -> 看最优配置是否翻正

直接调底层 solve_park_optimization 传 charge_cost，绕过未透传该参数的便捷函数，
不动 Q1/milp_model.py 与任何主模型/输出。
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_Q1_DIR = os.path.abspath(os.path.join(_HERE, '..', 'Q1'))
_Q2_DIR = _HERE
for _d in (_Q1_DIR, _Q2_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from data_loader import load_load_data, load_solar_wind_data, DT  # noqa: E402
from milp_model import solve_park_optimization  # noqa: E402
from q2_main import build_joint_profile  # noqa: E402

T = 24
P_STEP, E_STEP = 5, 10
P_MAX, E_MAX = 450, 1470        # 与主模型一致的上界
P_MAX_WIDE, E_MAX_WIDE = 450, 3000   # 放宽容量上界验证是否被截断


def _solve(charge_cost, integer=False, R0=None, P_max=P_MAX, E_max=E_MAX):
    kw = dict(
        optimize_capacity=True,
        capacity_bounds=(P_max, E_max),
        surplus_only_charge=True,
        charge_cost=charge_cost,   # None=View A(0.4/0.5)；(0,0)=View B(免费)
    )
    if integer:
        kw['capacity_steps'] = (P_STEP, E_STEP)
    if R0 is not None:
        kw['min_accom_rate'] = R0
    return solve_park_optimization(L_J, G_pv_J, G_w_J, **kw)


def report(name, r, P_max=P_MAX, E_max=E_MAX):
    if not r.get('success'):
        print(f"\n--- {name} ---  求解失败: {r.get('status')}")
        return
    ch = float(np.sum(r['P_ch']) * DT)
    dis = float(np.sum(r['P_dis']) * DT)
    hits = (r['P_ess'] >= P_max - 1e-3) or (r['E_ess'] >= E_max - 1e-3)
    print(f"\n--- {name} ---")
    print(f"  储能功率 P = {r['P_ess']:.3f} kW   容量 E = {r['E_ess']:.3f} kWh")
    print(f"  日运行成本 = {r['daily_cost']:.3f} 元   年化投资 = {r['inv_annual']:.2f} 元")
    print(f"  年实际成本 = {r['annual_cost']:.2f} 元")
    print(f"  日弃电 = {r['curt_total']:.3f} kWh   消纳率 = {r['re_ratio']*100:.4f}%")
    print(f"  日充电 = {ch:.3f} kWh   日放电 = {dis:.3f} kWh   日购电 = {r['grid_total']:.3f} kWh")
    print(f"  E/P 比 = {r['E_ess']/r['P_ess']:.3f} h" if r['P_ess'] > 1e-6 else "  E/P 比 = -")
    if hits:
        print(f"  ⚠ 触碰上界 {P_max}/{E_max}（需放宽验证）")


def main():
    global L_J, G_pv_J, G_w_J
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)

    print("=" * 68)
    print("View B 探索：充电免费（弃电机会成本=0），纯经济最优，不加消纳约束")
    print("=" * 68)

    # ---- 无储能基准（两口径相同：无充电）----
    r_base = solve_park_optimization(L_J, G_pv_J, G_w_J, P_ess=0, E_ess=0,
                                     optimize_capacity=False, surplus_only_charge=True)
    print(f"\n[基准] 联合无储能：日运行成本={r_base['daily_cost']:.3f}, "
          f"年实际={r_base['annual_cost']:.2f}, 弃电={r_base['curt_total']:.3f}, "
          f"消纳率={r_base['re_ratio']*100:.4f}%")

    # ---- View A（主模型口径，对比）----
    print("\n[View A] 充电按 0.4/0.5 计（主模型 ADR 0002）")
    report("View A 连续", _solve(charge_cost=None, integer=False))
    report("View A 工程整数", _solve(charge_cost=None, integer=True))

    # ---- View B（本探索）----
    print("\n[View B] 充电免费（弃电机会成本=0）")
    rB_c = _solve(charge_cost=(0.0, 0.0), integer=False)
    rB_i = _solve(charge_cost=(0.0, 0.0), integer=True)
    report("View B 连续", rB_c)
    report("View B 工程整数", rB_i)

    # ---- 若接近上界，放宽容量重跑（容差 2，避免粒度边界漏判）----
    hits_c = (rB_c['P_ess'] >= P_MAX - 2) or (rB_c['E_ess'] >= E_MAX - 2)
    if hits_c:
        print("\n[放宽] 容量上界 1470->3000、功率 450->1000 重跑，确认是否被上界截断")
        rB_wide_c = _solve(charge_cost=(0.0, 0.0), integer=False,
                          P_max=1000, E_max=3000)
        rB_wide_i = _solve(charge_cost=(0.0, 0.0), integer=True,
                          P_max=1000, E_max=3000)
        report("View B 连续(放宽上界)", rB_wide_c, 1000, 3000)
        report("View B 工程整数(放宽上界)", rB_wide_i, 1000, 3000)

    # ---- 边际经济验证：View B 下每 kWh 容量年净收益 ----
    print("\n[边际经济] View B 下首批储能每 kWh 容量年净收益估算：")
    eta_c, eta_d = 0.95, 0.95
    soc_util = 0.8
    # 日一充一放：每 kWh 容量 -> 放电 soc_util*eta_d kWh，替代购电 1.0 元
    ann_benefit_per_kwh = soc_util * eta_d * 365 * 1.0
    # View B 充电免费：充电成本 0
    # 年化投资：从 rB_i 取实际值
    inv_ann = rB_i['inv_annual']
    E_i = rB_i['E_ess']
    inv_per_kwh_ann = inv_ann / E_i if E_i > 1e-6 else 0.0
    net_per_kwh = ann_benefit_per_kwh - inv_per_kwh_ann
    print(f"  每 kWh 容量年放电收益 = {soc_util}×{eta_d}×365×1.0 = {ann_benefit_per_kwh:.1f} 元")
    print(f"  每 kWh 容量年化投资 = {inv_per_kwh_ann:.1f} 元（由工程解 {E_i:.0f} kWh 反算）")
    print(f"  每 kWh 容量年净 = {ann_benefit_per_kwh:.1f} − {inv_per_kwh_ann:.1f} = "
          f"{net_per_kwh:.1f} 元  {'> 0 → 配储有利可图' if net_per_kwh > 0 else '≤ 0 → 0/0'}")

    print("\n" + "=" * 68)
    print("结论对照")
    print("=" * 68)
    rA_i = _solve(charge_cost=None, integer=True)
    print(f"  View A 工程整数最优: P={rA_i['P_ess']:.0f} kW, E={rA_i['E_ess']:.0f} kWh, "
          f"年实际={rA_i['annual_cost']:.2f} 元, 消纳率={rA_i['re_ratio']*100:.3f}%")
    print(f"  View B 工程整数最优: P={rB_i['P_ess']:.0f} kW, E={rB_i['E_ess']:.0f} kWh, "
          f"年实际={rB_i['annual_cost']:.2f} 元, 消纳率={rB_i['re_ratio']*100:.3f}%")
    print(f"  基准(无储能)年实际: {r_base['annual_cost']:.2f} 元")
    save_b = r_base['annual_cost'] - rB_i['annual_cost']
    print(f"  View B 较无储能基准节省: {save_b:.2f} 元/年")


if __name__ == '__main__':
    main()
