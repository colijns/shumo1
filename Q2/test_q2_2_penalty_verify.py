# -*- coding: utf-8 -*-
"""问题2（2）带弃电惩罚（View C, ADR 0003）验收测试。

快速测试（默认）：<30s，覆盖：
  - λ=0 一致性：复现 View A 的 0/0=5514296.083、惩罚分解、c*/s 边界；
  - λ>0 惩罚适用：基准年综合 = 5514296.083+365·λ·1237.175、最优≤基准、
    最优弃电≤基准、消纳率≥基准、最优配置转正、低 c 下为正；
  - 结构性约束（λ=0.3 下 50/100）：功率平衡、不电网充电、余电充电、互斥。
慢测试（RUN_SLOW=1）：~5min，每个 λ 完整 2501 点网格验证网格最优=连续最优。

运行：
    python -m pytest Q2/test_q2_2_penalty_verify.py -v
    RUN_SLOW=1 python -m pytest Q2/test_q2_2_penalty_verify.py -v   # 含完整网格
"""

import os
import sys

import pytest

# 跨目录复用 Q2(2) 惩罚版实现
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from q2_storage_penalty import (  # noqa: E402
    build_joint_profile, load_load_data, load_solar_wind_data,
    solve_continuous, solve_integer, grid_search_2d,
    run_no_storage, run_fixed_storage,
    sensitivity_charge_cost, sensitivity_investment, sensitivity_lambda,
    LAMBDA_VALUES, P_STEP, E_STEP, T, TOL,
    JOINT_NO_STORAGE_ANNUAL, JOINT_NO_STORAGE_DAILY_CURT,
)
from grid_search import grid_search_capacity  # noqa: E402
from data_loader import C_P_ESS, C_E_ESS, Y  # noqa: E402

# 工程粒度取整引入的年化投资差上界：网格 P 步 5kW、E 步 10kWh，
# 连续最优 P 若为分数（如 109.04），网格取 110，差额 = ΔP·C_P_ESS/Y。
_GRANULARITY_BOUND = (P_STEP * C_P_ESS + E_STEP * C_E_ESS) / Y  # = 2200 元/年


ANN_TOL = 1.0  # 年综合成本容差 ±1 元
# 完整网格慢测试开关
_SLOW = bool(os.environ.get('RUN_SLOW'))


@pytest.fixture(scope='module')
def joint_profile():
    """读取附件并构造联合 profile（A/B/C 三园区聚合），module 级共享。"""
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L, Gpv, Gw = build_joint_profile(load_data, G_pv_data, G_w_data)
    assert len(L) == T and len(Gpv) == T and len(Gw) == T
    return L, Gpv, Gw


def _expected_baseline_annual(lam):
    """基准（无储能）年综合 = View A + 365·λ·1237.175（无储能弃电与 λ 无关）。"""
    return JOINT_NO_STORAGE_ANNUAL + 365 * lam * JOINT_NO_STORAGE_DAILY_CURT


# =====================================================================
# λ=0 一致性层（复现 View A）
# =====================================================================

def test_lambda_zero_reproduces_view_a(joint_profile):
    """λ=0 必须复现 View A：0/0、年综合 5514296.083、惩罚=0。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous(L, Gpv, Gw, lam=0.0)
    assert r['success']
    assert r['P_ess'] < 1e-4 and r['E_ess'] < 1e-4
    assert abs(r['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < ANN_TOL
    assert abs(r['daily_curt_penalty']) < TOL
    assert abs(r['daily_cost_pure'] - r['daily_cost']) < TOL


def test_lambda_zero_integer_zero(joint_profile):
    """λ=0 工程整数 MILP = 0/0。"""
    L, Gpv, Gw = joint_profile
    r = solve_integer(L, Gpv, Gw, lam=0.0)
    assert r['success']
    assert r['P_ess'] == 0.0 and r['E_ess'] == 0.0
    assert abs(r['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < ANN_TOL


def test_lambda_zero_charge_cost_break_even(joint_profile):
    """λ=0 充电价敏感性：c≤0.35 为正、c≥0.40 归零（盈亏平衡∈(0.35,0.40]）。"""
    L, Gpv, Gw = joint_profile
    rows = sensitivity_charge_cost(L, Gpv, Gw, lam=0.0)
    assert all((r['E_ess'] or 0) > 1e-3 for r in rows if r['c'] <= 0.35)
    assert all((r['E_ess'] or 0) < 1e-3 for r in rows if r['c'] >= 0.40)


def test_lambda_zero_investment_baseline(joint_profile):
    """λ=0 投资价 s=1 时容量为 0（小网格供 sensitivity_investment 复用 daily_cost）。"""
    L, Gpv, Gw = joint_profile
    # 小网格（5×5=25 点）足以验证 s 扫描逻辑，不跑完整 2501 点
    gs = grid_search_capacity(L, Gpv, Gw,
                              P_range=(0, 20, 5), E_range=(0, 40, 10),
                              verbose=False, curt_penalty=0.0)
    rows = sensitivity_investment(gs['grid'], lam=0.0)
    base = next(r for r in rows if r['s'] == 1.0)
    assert base['E_ess'] < 1e-3


# =====================================================================
# 基准惩罚数学（无储能弃电与 λ 无关 -> 年综合线性随 λ）
# =====================================================================

@pytest.mark.parametrize('lam', LAMBDA_VALUES)
def test_baseline_penalty_math(joint_profile, lam):
    """基准年综合 = 5514296.083 + 365·λ·1237.175；基准弃电恒为 1237.175。"""
    L, Gpv, Gw = joint_profile
    r_no = run_no_storage(L, Gpv, Gw, curt_penalty=lam)
    assert abs(r_no['annual_cost'] - _expected_baseline_annual(lam)) < ANN_TOL
    assert abs(r_no['curt_total'] - JOINT_NO_STORAGE_DAILY_CURT) < 1e-2
    # 惩罚分解
    assert abs(r_no['daily_cost'] - r_no['daily_cost_pure']
               - r_no['daily_curt_penalty']) < 1e-2
    assert abs(r_no['daily_curt_penalty'] - lam * r_no['curt_total']) < 1e-2


# =====================================================================
# λ>0 惩罚适用层
# =====================================================================

@pytest.mark.parametrize('lam', [0.3, 0.6])
def test_penalty_flips_storage_positive(joint_profile, lam):
    """λ>0 时最优配置从 0/0 翻转为正（惩罚使弃电消纳边际收益增加）。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous(L, Gpv, Gw, lam=lam)
    assert r['success']
    assert r['E_ess'] > 1e-3, f"λ={lam} 期望 E_ess>0，实际 {r['E_ess']}"
    assert r['P_ess'] > 1e-3


@pytest.mark.parametrize('lam', [0.3, 0.6])
def test_optimal_dominates_baseline(joint_profile, lam):
    """最优年综合 ≤ 基准；最优弃电 ≤ 基准弃电；消纳率 ≥ 基准。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous(L, Gpv, Gw, lam=lam)
    r_no = run_no_storage(L, Gpv, Gw, curt_penalty=lam)
    assert r['annual_cost'] <= r_no['annual_cost'] + ANN_TOL
    assert r['curt_total'] <= r_no['curt_total'] + 1e-2
    assert r['re_ratio'] >= r_no['re_ratio'] - 1e-6
    # 严格更优：储能实际被采用 -> 年综合严格低于基准
    assert r['annual_cost'] < r_no['annual_cost'] - 1.0


@pytest.mark.parametrize('lam', [0.3, 0.6])
def test_low_charge_cost_positive(joint_profile, lam):
    """固定 λ>0、c=0（免费充电）下容量为正。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous(L, Gpv, Gw, lam=lam, charge_cost=(0.0, 0.0))
    assert r['success']
    assert r['E_ess'] > 1e-3


# =====================================================================
# 结构性约束（λ=0.3 下 50/100 验证有储能时的约束仍成立）
# =====================================================================

def test_structural_constraints_at_lambda_03(joint_profile):
    """λ=0.3 下 50/100：功率平衡、不电网充电、余电充电、互斥、SOC。"""
    L, Gpv, Gw = joint_profile
    r = run_fixed_storage(L, Gpv, Gw, P_ess=50, E_ess=100, curt_penalty=0.3)
    assert r['errors']['max_balance_error'] < TOL
    assert r['errors']['max_re_error'] < TOL
    for t in range(T):
        # 不电网充电
        assert abs(r['P_ch'][t] - r['P_ch_pv'][t] - r['P_ch_w'][t]) < 1e-6
        # 互斥
        assert not (r['P_ch'][t] > 1e-4 and r['P_dis'][t] > 1e-4)
        # 余电充电
        if r['P_ch'][t] > 1e-4:
            assert Gpv[t] + Gw[t] - L[t] > -TOL
    # 惩罚分解
    assert abs(r['daily_cost'] - r['daily_cost_pure'] - r['daily_curt_penalty']) < 1e-2


# =====================================================================
# λ 敏感性主线
# =====================================================================

def test_lambda_sensitivity_structure(joint_profile):
    """sensitivity_lambda 返回 3 行；弃电随 λ 非递增；消纳率非递减。"""
    L, Gpv, Gw = joint_profile
    rows = sensitivity_lambda(L, Gpv, Gw, LAMBDA_VALUES)
    assert len(rows) == len(LAMBDA_VALUES)
    curts = [r['curt_total'] for r in rows]
    rerat = [r['re_ratio'] for r in rows]
    # λ=0 基准弃电
    assert abs(curts[0] - JOINT_NO_STORAGE_DAILY_CURT) < 1e-2
    # 单调
    for i in range(len(curts) - 1):
        assert curts[i] >= curts[i + 1] - 1e-2, "弃电应随 λ 非递增"
        assert rerat[i] <= rerat[i + 1] + 1e-6, "消纳率应随 λ 非递减"
    # 每行节省 ≥ 0
    assert all(r['annual_saving'] >= -ANN_TOL for r in rows)


def test_lambda_saving_grows_with_lambda(joint_profile):
    """λ 越大，储能相对基准的年综合节省越多（惩罚放大储能价值）。"""
    L, Gpv, Gw = joint_profile
    rows = sensitivity_lambda(L, Gpv, Gw, LAMBDA_VALUES)
    saves = [r['annual_saving'] for r in rows]
    # λ=0 节省 ≈ 0（0/0 最优）；λ=0.6 节省 > λ=0.3 节省 > λ=0
    assert saves[1] > saves[0] + 1.0
    assert saves[2] > saves[1]


# =====================================================================
# 完整网格（慢，默认跳过；RUN_SLOW=1 启用）
# =====================================================================

@pytest.mark.skipif(not _SLOW, reason="完整 2501 点网格×3 耗时 ~5min，设 RUN_SLOW=1 启用")
@pytest.mark.parametrize('lam', LAMBDA_VALUES)
def test_grid_matches_continuous(joint_profile, lam):
    """完整 2D 网格全局最优 = 连续 MILP 最优（±工程粒度）。"""
    L, Gpv, Gw = joint_profile
    r_cont = solve_continuous(L, Gpv, Gw, lam=lam)
    gs = grid_search_2d(L, Gpv, Gw, lam=lam, verbose=False)
    best = gs['best']
    assert abs(best['P_ess'] - r_cont['P_ess']) <= P_STEP + TOL
    assert abs(best['E_ess'] - r_cont['E_ess']) <= E_STEP + TOL
    # 网格成本 ≥ 连续成本（连续是下界），且差距在工程粒度内。
    # 连续最优 P 若为分数（如 λ=0.3 时 109.04），网格取 110，差额 = ΔP·C_P_ESS/Y
    # 纯属功率粒度的年化投资差（E=600 两边一致），非逻辑错误。
    assert best['annual_cost'] >= r_cont['annual_cost'] - ANN_TOL
    assert abs(best['annual_cost'] - r_cont['annual_cost']) <= _GRANULARITY_BOUND + ANN_TOL


@pytest.mark.skipif(not _SLOW, reason="完整网格慢测试，设 RUN_SLOW=1 启用")
def test_lambda_zero_grid_origin_optimal(joint_profile):
    """λ=0 完整网格：0/0 为全局最优（复现无惩罚版结论）。"""
    L, Gpv, Gw = joint_profile
    gs = grid_search_2d(L, Gpv, Gw, lam=0.0, verbose=False)
    best = gs['best']
    assert best['P_ess'] == 0 and best['E_ess'] == 0
    assert abs(best['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < ANN_TOL
