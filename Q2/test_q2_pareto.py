# -*- coding: utf-8 -*-
"""问题2（2）Pareto 主模型（ADR 0004）验收测试。

覆盖 docs/问题2pareto.md §2.2.12 的 9 项自动化检查 + 推荐配置关键数字 +
Pareto 前沿关键点 + ε-约束向后兼容性。所有断言数字均来自 Q2/q2_pareto.py
真实求解器输出（已与问题2pareto.md 预测值逐项核对一致）。

快速测试（默认）：<30s，覆盖：
  - §2.2.12 九项自动化检查全部通过；
  - 推荐配置 = 175 kW / 890 kWh（R0*=97%），实际消纳率 97.019%、年实际成本
    5,575,300.76 元、较 0/0 基准 +61,004.68 元；
  - Pareto 前沿关键点（连续 + 工程整数）的精确值；
  - 等权理想点距离：97% 档 d=0.659 最小、基准与 100% 档 d=1.000；
  - ε-约束向后兼容：R0=None 不加约束，复现 Q2.1 纯经济基准 0/0。

运行：
    python -m pytest Q2/test_q2_pareto.py -v
"""

import os
import sys

import pytest

# 跨目录复用 Q2(2) Pareto 主模型实现
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from q2_pareto import (  # noqa: E402
    build_joint_profile, load_load_data, load_solar_wind_data,
    solve_continuous_at, solve_integer_at,
    build_pareto_front, ideal_point_selection, verify_pareto,
    R0_TARGETS, P_MAX, E_MAX, P_STEP, E_STEP, T, TOL,
    JOINT_NO_STORAGE_ANNUAL, JOINT_NO_STORAGE_RE_RATIO,
)

ANN_TOL = 1.0   # 年成本容差 ±1 元（CBC 求解器精度余量）
RE_TOL = 1e-3   # 消纳率容差（百分比显示精度）


@pytest.fixture(scope='module')
def joint_profile():
    """读取附件并构造联合 profile（A/B/C 三园区聚合），module 级共享。"""
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    L, Gpv, Gw = build_joint_profile(load_data, G_pv_data, G_w_data)
    assert len(L) == T and len(Gpv) == T and len(Gw) == T
    return L, Gpv, Gw


@pytest.fixture(scope='module')
def pareto_front(joint_profile):
    """构造完整 Pareto 前沿（连续 + 工程整数 + 基准），module 级共享。约 15s。"""
    L, Gpv, Gw = joint_profile
    return build_pareto_front(L, Gpv, Gw, verbose=False)


@pytest.fixture(scope='module')
def selection(pareto_front):
    """理想点距离选型结果，module 级共享。"""
    return ideal_point_selection(pareto_front)


@pytest.fixture(scope='module')
def checks(joint_profile, pareto_front, selection):
    """§2.2.12 九项自动化检查结果（仅求解一次，含第 9 项宽上界复算）。"""
    L, Gpv, Gw = joint_profile
    return verify_pareto(pareto_front, selection, L, Gpv, Gw)


# =====================================================================
# §2.2.12 九项自动化检查
# =====================================================================

def test_verify_has_nine_checks(checks):
    """verify_pareto 返回恰好 9 项检查。"""
    assert len(checks) == 9


def test_check1_all_schemes_meet_target_accommodation(checks):
    """检查1：各档工程整数方案实际消纳率 ≥ 设定 R0。"""
    name, ok, detail = checks[0]
    assert name.startswith('1.')
    assert ok, detail


def test_check2_curt_nonincr_cost_nondecr(checks):
    """检查2：消纳目标越严，弃电不增、成本不降（Pareto 单调性）。"""
    name, ok, detail = checks[1]
    assert name.startswith('2.')
    assert ok, detail


def test_check3_engineering_granularity(checks):
    """检查3：工程容量满足 5kW/10kWh 粒度。"""
    name, ok, detail = checks[2]
    assert name.startswith('3.')
    assert ok, detail


def test_check4_balance_and_allocation_error(checks):
    """检查4：功率平衡/风光分配最大误差 ≤ 1e-5。"""
    name, ok, detail = checks[3]
    assert name.startswith('4.')
    assert ok, detail


def test_check5_surplus_only_charge(checks):
    """检查5：储能只用风光余电充电（P_ch ≤ max(G_pv+G_w−L, 0)）。"""
    name, ok, detail = checks[4]
    assert name.startswith('5.')
    assert ok, detail


def test_check6_mutual_exclusion_soc_bounds(checks):
    """检查6：充放电互斥、SOC∈[0.1,0.9]、初末 SOC 相等。"""
    name, ok, detail = checks[5]
    assert name.startswith('6.')
    assert ok, detail


def test_check7_accommodation_curtailment_equivalence(checks):
    """检查7：消纳率约束与弃电率约束等价（re_ratio≥R0 ⇔ E_curt≤(1−R0)·E_re）。"""
    name, ok, detail = checks[6]
    assert name.startswith('7.')
    assert ok, detail


def test_check8_zero_curtailment_full_accommodation(checks):
    """检查8：R0=100% 方案实现零弃电、100% 消纳。"""
    name, ok, detail = checks[7]
    assert name.startswith('8.')
    assert ok, detail


def test_check9_wide_bound_unchanged(checks):
    """检查9：上界放宽到 550/1800 后，零弃电工程最优仍为 450/1470。"""
    name, ok, detail = checks[8]
    assert name.startswith('9.')
    assert ok, detail


# =====================================================================
# 推荐配置关键数字（§2.2.9 / §2.2.11）
# =====================================================================

def test_recommended_R0_is_97_percent(selection):
    """理想点距离选型推荐目标 R0* = 97%。"""
    assert selection['recommended_R0'] == 0.97


def test_recommended_integer_capacity(selection):
    """推荐工程整数配置 = 175 kW / 890 kWh。"""
    r = selection['recommended_int']
    assert r['P_ess'] == 175.0
    assert r['E_ess'] == 890.0


def test_recommended_accommodation_and_cost(selection):
    """推荐方案实际消纳率 97.019%、年实际成本 5,575,300.76 元。"""
    r = selection['recommended_int']
    assert abs(r['re_ratio'] - 0.97019) < RE_TOL
    assert abs(r['annual_cost'] - 5575300.76) < ANN_TOL


def test_recommended_increment_over_baseline(pareto_front, selection):
    """推荐方案较 0/0 基准年实际成本增加 +61,004.68 元。"""
    base = pareto_front['baseline_cont']['annual_cost']
    rec = selection['recommended_int']['annual_cost']
    assert abs((rec - base) - 61004.68) < ANN_TOL


# =====================================================================
# Pareto 前沿关键点（§2.2.9 连续表 + 工程整数表）
# =====================================================================

def test_baseline_zero_storage(pareto_front):
    """纯经济基准（R0=None）= 0/0，复现 Q2.1 联合无储能结论。"""
    r = pareto_front['baseline_cont']
    assert r['P_ess'] < TOL and r['E_ess'] < TOL
    assert abs(r['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < ANN_TOL
    assert abs(r['re_ratio'] - JOINT_NO_STORAGE_RE_RATIO) < RE_TOL


def test_continuous_front_key_points(pareto_front):
    """连续 Pareto 前沿 5 档关键点（P/E/年实际成本）。"""
    expected = {
        0.95: (87.549, 497.895, 5546235.32),
        0.97: (173.964, 886.395, 5575015.00),
        0.98: (238.091, 1080.645, 5591078.35),
        0.99: (319.880, 1274.895, 5608554.73),
        1.00: (448.115, 1469.145, 5629746.70),
    }
    for R0, (P, E, C) in expected.items():
        r = pareto_front['by_R0'][R0]['cont']
        assert abs(r['P_ess'] - P) < 1e-2, f"R0={R0} P={r['P_ess']}"
        assert abs(r['E_ess'] - E) < 1e-2, f"R0={R0} E={r['E_ess']}"
        assert abs(r['annual_cost'] - C) < ANN_TOL, f"R0={R0} C={r['annual_cost']}"


def test_integer_front_key_points(pareto_front):
    """工程整数 Pareto 前沿 5 档关键点（P/E/年实际成本/消纳率）。"""
    expected = {
        0.95: (90, 500, 5546549.92, 0.95011),
        0.97: (175, 890, 5575300.76, 0.97019),
        0.98: (245, 1090, 5592157.61, 0.98048),
        0.99: (325, 1280, 5609251.61, 0.99026),
        1.00: (450, 1470, 5630051.34, 1.00000),
    }
    for R0, (P, E, C, R) in expected.items():
        r = pareto_front['by_R0'][R0]['int']
        assert r['P_ess'] == P, f"R0={R0} P={r['P_ess']}"
        assert r['E_ess'] == E, f"R0={R0} E={r['E_ess']}"
        assert abs(r['annual_cost'] - C) < ANN_TOL, f"R0={R0} C={r['annual_cost']}"
        assert abs(r['re_ratio'] - R) < RE_TOL, f"R0={R0} R={r['re_ratio']}"


def test_integer_granularity_all_targets(pareto_front):
    """工程整数各档 P%5=0、E%10=0。"""
    for R0 in R0_TARGETS:
        r = pareto_front['by_R0'][R0]['int']
        assert abs(r['P_ess'] % P_STEP) < 1e-6, f"R0={R0} P={r['P_ess']}"
        assert abs(r['E_ess'] % E_STEP) < 1e-6, f"R0={R0} E={r['E_ess']}"


def test_zero_curtailment_at_100_percent(pareto_front):
    """R0=100% 工程整数方案实现零弃电。"""
    r = pareto_front['by_R0'][1.00]['int']
    assert abs(r['curt_total']) < TOL
    assert abs(r['re_ratio'] - 1.0) < TOL


# =====================================================================
# 等权理想点距离选型（§2.2.9 理想点距离表）
# =====================================================================

def test_ideal_point_min_distance_at_97(selection):
    """97% 档理想点距离最小（d≈0.659），基准与 100% 档 d=1.000。"""
    cands = {c['label']: c for c in selection['candidates']}
    assert abs(cands['97%目标']['ideal_dist'] - 0.659) < 1e-2
    assert abs(cands['纯经济基准']['ideal_dist'] - 1.000) < 1e-3
    assert abs(cands['100%目标']['ideal_dist'] - 1.000) < 1e-3
    # best 即 97% 档
    assert selection['best']['R0'] == 0.97


def test_ideal_point_distances_monotone_arms(selection):
    """理想点距离呈 U 形：97% 最低，向两侧（基准、100%）递增。"""
    d = {c['R0']: c['ideal_dist'] for c in selection['candidates']}
    # 97% 严格小于其它所有候选档（含基准 None 与 100%）
    assert d[0.97] < d[0.95]
    assert d[0.97] < d[0.98]
    assert d[0.97] < d[0.99]
    assert d[0.97] < d[None]   # 基准
    assert d[0.97] < d[1.00]


# =====================================================================
# ε-约束向后兼容性（R0=None 不加约束）
# =====================================================================

def test_epsilon_constraint_backward_compatible(joint_profile):
    """R0=None 时不加消纳率约束，等价于 Q2.1 纯经济最优 0/0。"""
    L, Gpv, Gw = joint_profile
    r_none = solve_continuous_at(L, Gpv, Gw, R0=None)
    assert r_none['success']
    assert r_none['P_ess'] < TOL and r_none['E_ess'] < TOL
    assert abs(r_none['annual_cost'] - JOINT_NO_STORAGE_ANNUAL) < ANN_TOL
    assert r_none['min_accom_rate'] is None


def test_epsilon_constraint_active_when_set(joint_profile):
    """R0=0.97 时约束生效，结果 min_accom_rate=0.97 且 re_ratio≥0.97。"""
    L, Gpv, Gw = joint_profile
    r = solve_continuous_at(L, Gpv, Gw, R0=0.97)
    assert r['success']
    assert r['min_accom_rate'] == 0.97
    assert r['re_ratio'] >= 0.97 - TOL
    # 约束生效使成本高于纯经济基准
    assert r['annual_cost'] > JOINT_NO_STORAGE_ANNUAL


# =====================================================================
# Pareto 单调性（成本递增、弃电递减）
# =====================================================================

def test_pareto_monotonicity(pareto_front):
    """消纳目标越严：年实际成本单调非降、弃电量单调非增。"""
    costs = [pareto_front['by_R0'][R0]['int']['annual_cost'] for R0 in R0_TARGETS]
    curts = [pareto_front['by_R0'][R0]['int']['curt_total'] for R0 in R0_TARGETS]
    assert all(costs[i + 1] >= costs[i] - 1e-3 for i in range(len(costs) - 1))
    assert all(curts[i + 1] <= curts[i] + 1e-6 for i in range(len(curts) - 1))
