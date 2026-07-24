# -*- coding: utf-8 -*-
"""问题2（2）验收测试：覆盖 docs/问题2（2）.md §6 关键数值与 §9 检查项。

快速测试（默认）：<15s，涵盖连续/整数 MILP、固定 50/100、结构性约束、
                  敏感性临界点、小网格 0/0 局部最优。
慢测试（RUN_SLOW=1）：~90s，完整 2501 点网格验证 0/0 全局最优与最优正配置 5/10。

运行：
    python -m pytest Q2/test_q2_2_verify.py -v
    RUN_SLOW=1 python -m pytest Q2/test_q2_2_verify.py -v   # 含完整网格
"""

import os
import sys

import pytest

# 跨目录复用 Q2(2) 实现（q2_storage 顶部会自行把 Q1/Q2/templates 加入 sys.path）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from q2_storage import (  # noqa: E402
    build_joint_profile, load_load_data, load_solar_wind_data,
    solve_continuous, solve_integer, grid_search_2d,
    run_no_storage, run_fixed_storage,
    sensitivity_charge_cost,
    P_STEP, E_STEP, T, TOL,
)
from grid_search import grid_search_capacity  # noqa: E402


# ---- §6 期望数值（来自完整运行 _run_log.txt，作为回归基线） ----
EXPECTED = {
    'no_storage_annual': 5514296.083,        # 联合无储能 == Q2(1) 基准
    'no_storage_daily': 15107.661,
    'cont_P': 0.0, 'cont_E': 0.0,
    'cont_annual': 5514296.083,              # 连续 MILP 退化为 0/0
    'int_P': 0.0, 'int_E': 0.0,
    'int_annual': 5514296.083,               # 工程整数 MILP 退化为 0/0
    'fixed_50_100_annual': 5521826.305,      # 50/100：日运行省、年投资 22000、净亏
    'fixed_50_100_inv_annual': 22000.0,
    'best_positive_P': 5.0, 'best_positive_E': 10.0,
    'best_positive_annual': 5514951.556,     # 最优正配置仍劣于 0/0
}
ANN_TOL = 1.0  # 年综合成本容差 ±1 元


@pytest.fixture(scope='module')
def joint_profile():
    """读取附件并构造联合 profile（A/B/C 三园区聚合），module 级共享。"""
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L, Gpv, Gw = build_joint_profile(load_data, G_pv_data, G_w_data)
    assert len(L) == T and len(Gpv) == T and len(Gw) == T
    return L, Gpv, Gw


# =====================================================================
# §6 数值一致性
# =====================================================================

def test_no_storage_baseline(joint_profile):
    """联合无储能年综合 = Q2(1) 基准 5514296.083。"""
    L, Gpv, Gw = joint_profile
    r = run_no_storage(L, Gpv, Gw)
    assert abs(r['annual_cost'] - EXPECTED['no_storage_annual']) < ANN_TOL
    assert abs(r['daily_cost'] - EXPECTED['no_storage_daily']) < 1e-2


def test_continuous_milp_zero(joint_profile):
    """连续 MILP 理论最优 = 0/0。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous(L, Gpv, Gw)
    assert r['success']
    assert r['P_ess'] < 1e-4 and r['E_ess'] < 1e-4
    assert abs(r['annual_cost'] - EXPECTED['cont_annual']) < ANN_TOL


def test_integer_milp_zero(joint_profile):
    """工程整数 MILP (5kW/10kWh 粒度) = 0/0。"""
    L, Gpv, Gw = joint_profile
    r = solve_integer(L, Gpv, Gw)
    assert r['success']
    assert r['P_ess'] == 0.0 and r['E_ess'] == 0.0
    assert abs(r['annual_cost'] - EXPECTED['int_annual']) < ANN_TOL


def test_fixed_50_100(joint_profile):
    """固定 50kW/100kWh：年综合 5521826.305，年均投资 22000。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    assert abs(r['annual_cost'] - EXPECTED['fixed_50_100_annual']) < ANN_TOL
    assert abs(r['inv_annual'] - EXPECTED['fixed_50_100_inv_annual']) < 1e-2


# =====================================================================
# §9 结构性检查（用 50/100 正配置验证有储能时的约束）
# =====================================================================

def test_power_balance(joint_profile):
    """每小时功率平衡误差 < 1e-4。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    assert r['errors']['max_balance_error'] < TOL


def test_re_dispatch(joint_profile):
    """每小时风光出力分配误差 < 1e-4。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    assert r['errors']['max_re_error'] < TOL


def test_no_grid_charging(joint_profile):
    """结构性禁止电网充电：充电功率 == 光伏充电 + 风电充电。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    for t in range(T):
        assert abs(r['P_ch'][t] - r['P_ch_pv'][t] - r['P_ch_w'][t]) < 1e-6


def test_surplus_only_charging(joint_profile):
    """余电充电涌现：充电仅出现在风光有富余的时段（缺电时段不充电）。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    charged_hours = sum(1 for t in range(T) if r['P_ch'][t] > 1e-4)
    assert charged_hours > 0, "50/100 应出现充电"
    for t in range(T):
        if r['P_ch'][t] > 1e-4:
            surplus = Gpv[t] + Gw[t] - L[t]
            assert surplus > -TOL, f"t={t} 缺电时段充电 {r['P_ch'][t]:.2f}kW"


def test_no_simultaneous_charge_discharge(joint_profile):
    """不同时充放电（z 互斥约束）。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    for t in range(T):
        assert not (r['P_ch'][t] > 1e-4 and r['P_dis'][t] > 1e-4), f"t={t} 同时充放电"


def test_soc_bounds_and_periodicity(joint_profile):
    """SOC ∈ [10%, 90%]，日末 SOC = 日初。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    e = r['errors']
    assert e['min_soc'] >= 0.10 - TOL
    assert e['max_soc'] <= 0.90 + TOL
    assert e['end_soc_error'] < TOL


def test_annual_cost_decomposition(joint_profile):
    """年综合 = 365 × 日运行 + 年均投资。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100)
    recon = 365 * r['daily_cost'] + r['inv_annual']
    assert abs(recon - r['annual_cost']) < 1e-2


# =====================================================================
# §9 工程粒度
# =====================================================================

def test_engineering_granularity(joint_profile):
    """整数解 P 是 5kW 倍数、E 是 10kWh 倍数。"""
    L, Gpv, Gw = joint_profile
    r = solve_integer(L, Gpv, Gw)
    assert abs(r['P_ess'] - round(r['P_ess'] / P_STEP) * P_STEP) < 1e-6
    assert abs(r['E_ess'] - round(r['E_ess'] / E_STEP) * E_STEP) < 1e-6


# =====================================================================
# 敏感性：盈亏平衡点
# =====================================================================

def test_charge_cost_break_even(joint_profile):
    """充电价 c≤0.35 容量为正，c≥0.40 归零（盈亏平衡 ∈ (0.35,0.40]）。"""
    L, Gpv, Gw = joint_profile
    sens = sensitivity_charge_cost(
        L, Gpv, Gw, c_values=[0.0, 0.20, 0.35, 0.40, 0.50])
    by_c = {round(r['c'], 2): r for r in sens}
    assert by_c[0.0]['E_ess'] > 1e-3, "c=0 应建储能"
    assert by_c[0.20]['E_ess'] > 1e-3
    assert by_c[0.35]['E_ess'] > 1e-3, "c=0.35 仍为正"
    assert by_c[0.40]['E_ess'] < 1e-3, "c=0.40 归零（光伏价边界）"
    assert by_c[0.50]['E_ess'] < 1e-3, "c=0.50 归零（风电价）"


def test_charge_cost_monotonic_capacity(joint_profile):
    """充电价越低、最优容量越大（经济性单调）。"""
    L, Gpv, Gw = joint_profile
    sens = sensitivity_charge_cost(
        L, Gpv, Gw, c_values=[0.0, 0.1, 0.2, 0.3])
    caps = [r['E_ess'] for r in sens]
    for i in range(len(caps) - 1):
        assert caps[i] >= caps[i + 1] - 1e-3


# =====================================================================
# 小网格：0/0 局部最优（快速，25 点）
# =====================================================================

def test_small_grid_zero_local_optimum(joint_profile):
    """小范围网格 (P∈[0,20]/5, E∈[0,40]/10, 25 点) 验证 0/0 为局部最优。"""
    L, Gpv, Gw = joint_profile
    gs = grid_search_capacity(
        L, Gpv, Gw,
        P_range=(0, 20, P_STEP), E_range=(0, 40, E_STEP),
        verbose=False)
    best = gs['best']
    assert best['P_ess'] == 0 and best['E_ess'] == 0
    # 原点成本应 ≤ 任一邻居
    origin = next(p for p in gs['grid']
                  if p['P_ess'] == 0 and p['E_ess'] == 0)
    for p in gs['grid']:
        assert origin['annual_cost'] <= p['annual_cost'] + 1e-6


# =====================================================================
# 完整网格（慢，默认跳过；RUN_SLOW=1 启用）
# =====================================================================

_SLOW = bool(os.environ.get('RUN_SLOW'))


@pytest.mark.skipif(not _SLOW, reason="完整 2501 点网格耗时 ~90s，设 RUN_SLOW=1 启用")
def test_full_grid_zero_global_optimum(joint_profile):
    """完整 2501 点网格：0/0 全局最优，最优正配置 5/10。"""
    L, Gpv, Gw = joint_profile
    gs = grid_search_2d(L, Gpv, Gw, verbose=False)
    best = gs['best']
    assert best['P_ess'] == 0 and best['E_ess'] == 0
    assert abs(best['annual_cost'] - EXPECTED['no_storage_annual']) < ANN_TOL
    bp = gs['best_positive']
    assert bp is not None
    assert bp['P_ess'] == EXPECTED['best_positive_P']
    assert bp['E_ess'] == EXPECTED['best_positive_E']
    assert abs(bp['annual_cost'] - EXPECTED['best_positive_annual']) < ANN_TOL
    # 最优正配置仍劣于 0/0
    assert bp['annual_cost'] > best['annual_cost']
