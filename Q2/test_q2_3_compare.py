# -*- coding: utf-8 -*-
"""问题2（3）三层对比（ADR 0004）验收测试。

覆盖 docs/问题2pareto.md §2.3 的三层结果关键数字 + 结构性约束：
  - 独立纯经济工程最优 = Q1 的 95/190（A=0/0, B=35/120, C=60/70）；
  - 独立各园区 ≥97% 工程配置 = A=160/990, B=165/750, C=190/1050（合计 515/2790）；
  - 联合 Pareto 推荐 = 175/890，联合纯经济 = 0/0；
  - 三层收益：B_final=301,329.25(r=5.13%)、B_economic=362,333.93(r=6.17%)、
    B_97%=440,091.57(r=7.32%)；
  - 独立 ≥97% 上界不 binding、各园区实际消纳率 ≥97%。

运行：
    python -m pytest Q2/test_q2_3_compare.py -v
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from q2_3_compare import (  # noqa: E402
    run_q2_3_compare, PARKS, P_MAX_97, E_MAX_97,
)

ANN_TOL = 1.0   # 年成本容差 ±1 元
RE_TOL = 1e-3   # 消纳率容差


@pytest.fixture(scope='module')
def comparison(tmp_path_factory):
    """运行一次三层对比主流程，module 级共享。约 20s。"""
    out = tmp_path_factory.mktemp('q2_3_out')
    return run_q2_3_compare(output_dir=str(out))


# =====================================================================
# 独立纯经济工程最优（复现 Q1 的 95/190）
# =====================================================================

def test_independent_pure_economic_capacity(comparison):
    """独立纯经济工程最优：A=0/0, B=35/120, C=60/70，合计 95/190。"""
    res = comparison['ind_pure_res']
    assert res['A']['P_ess'] == 0 and res['A']['E_ess'] == 0
    assert res['B']['P_ess'] == 35 and res['B']['E_ess'] == 120
    assert res['C']['P_ess'] == 60 and res['C']['E_ess'] == 70


def test_independent_pure_economic_annual_cost(comparison):
    """独立纯经济各园区年实际成本与合计。"""
    res = comparison['ind_pure_res']
    assert abs(res['A']['annual_cost'] - 2220979.38) < ANN_TOL
    assert abs(res['B']['annual_cost'] - 1846084.92) < ANN_TOL
    assert abs(res['C']['annual_cost'] - 1809565.72) < ANN_TOL
    total = sum(res[p]['annual_cost'] for p in PARKS)
    assert abs(total - 5876630.02) < ANN_TOL


# =====================================================================
# 独立各园区 ≥97% 工程最优
# =====================================================================

def test_independent_97_capacity(comparison):
    """独立 ≥97%：A=160/990, B=165/750, C=190/1050，合计 515/2790。"""
    res = comparison['ind_97_res']
    assert res['A']['P_ess'] == 160 and res['A']['E_ess'] == 990
    assert res['B']['P_ess'] == 165 and res['B']['E_ess'] == 750
    assert res['C']['P_ess'] == 190 and res['C']['E_ess'] == 1050
    assert sum(res[p]['P_ess'] for p in PARKS) == 515
    assert sum(res[p]['E_ess'] for p in PARKS) == 2790


def test_independent_97_meets_target(comparison):
    """独立 ≥97% 各园区实际消纳率 ≥ 97%。"""
    res = comparison['ind_97_res']
    for p in PARKS:
        assert res[p]['re_ratio'] >= 0.97 - 1e-6, f"园区{p} 未达 97%"


def test_independent_97_accommodation_rates(comparison):
    """独立 ≥97% 各园区实际消纳率精确值。"""
    res = comparison['ind_97_res']
    assert abs(res['A']['re_ratio'] - 0.97046) < RE_TOL
    assert abs(res['B']['re_ratio'] - 0.97000) < RE_TOL
    assert abs(res['C']['re_ratio'] - 0.97017) < RE_TOL


def test_independent_97_bounds_not_binding(comparison):
    """独立 ≥97% 各园区最优容量严格小于上界 500/1500（不 binding）。"""
    res = comparison['ind_97_res']
    for p in PARKS:
        assert res[p]['P_ess'] < P_MAX_97, f"园区{p} P 触上界"
        assert res[p]['E_ess'] < E_MAX_97, f"园区{p} E 触上界"


# =====================================================================
# 联合侧
# =====================================================================

def test_joint_97_capacity(comparison):
    """联合 ≥97% Pareto 推荐 = 175/890。"""
    r = comparison['r_jnt_97']
    assert r['P_ess'] == 175 and r['E_ess'] == 890


def test_joint_97_metrics(comparison):
    """联合 97% 方案消纳率与年实际成本。"""
    r = comparison['r_jnt_97']
    assert abs(r['re_ratio'] - 0.97019) < RE_TOL
    assert abs(r['annual_cost'] - 5575300.76) < ANN_TOL


def test_joint_pure_economic_zero(comparison):
    """联合纯经济最优 = 0/0，复现 Q2.1 联合无储能结论。"""
    r = comparison['r_jnt_pure']
    assert r['P_ess'] < 1e-4 and r['E_ess'] < 1e-4
    assert abs(r['annual_cost'] - 5514296.08) < ANN_TOL


# =====================================================================
# 三层收益
# =====================================================================

def test_benefit_final(comparison):
    """§2.3.2 最终方案收益 B=301,329.25, r=5.13%。"""
    B, r = comparison['benefit_final']
    assert abs(B - 301329.25) < ANN_TOL
    assert abs(r - 5.13) < 0.05


def test_benefit_economic(comparison):
    """§2.3.3 纯经济收益 B=362,333.93, r=6.17%。"""
    B, r = comparison['benefit_economic']
    assert abs(B - 362333.93) < ANN_TOL
    assert abs(r - 6.17) < 0.05


def test_benefit_97(comparison):
    """§2.3.4 同97%收益 B=440,091.57, r=7.32%。"""
    B, r = comparison['benefit_97']
    assert abs(B - 440091.57) < ANN_TOL
    assert abs(r - 7.32) < 0.05


def test_benefit_monotone_three_layers(comparison):
    """三层收益单调：B_97% > B_economic > B_final。

    - 同97%：独立侧各园区需大储能(515/2790)，联合仅需175/890，省得最多；
    - 纯经济：两侧均无绿色要求，联合省储能投资+运行成本；
    - 最终方案：独立侧纯经济(低成本)、联合侧97%(高成本)，差距最小。
    """
    Bf = comparison['benefit_final'][0]
    Be = comparison['benefit_economic'][0]
    B9 = comparison['benefit_97'][0]
    assert B9 > Be > Bf


# =====================================================================
# 对比表结构
# =====================================================================

def test_comparison_tables_have_expected_rows(comparison):
    """三张对比表均含 10 个指标行（含风光消纳率）。"""
    for name in ('tbl_final', 'tbl_econ', 'tbl_97'):
        df = comparison[name]
        assert len(df) == 10, f"{name} 行数={len(df)}"
        assert '年实际成本/元' in df['指标'].values
        assert '风光消纳率' in df['指标'].values


def test_independent_97_detail_table(comparison):
    """独立 ≥97% 明细表含 A/B/C + 合计 4 行。"""
    df = comparison['det_97']
    assert len(df) == 4
    assert list(df['园区']) == ['A', 'B', 'C', '合计']
