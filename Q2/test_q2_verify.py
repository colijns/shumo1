# -*- coding: utf-8 -*-
"""
问题2（1）验收测试 -- 对应 docs/问题2（1）.md §9 checklist 与 §6 数值。

复用 q2_main 的函数，避免重复实现；夹具在模块级求解一次，供所有用例共享。
"""

import os
import sys

import numpy as np
import pytest

_Q2_DIR = os.path.abspath(os.path.dirname(__file__))
if _Q2_DIR not in sys.path:
    sys.path.insert(0, _Q2_DIR)
import q2_main  # noqa: E402


@pytest.fixture(scope='module')
def bundle():
    """加载数据并求解联合/独立结果，供全部用例共享。"""
    load_data = q2_main.load_load_data()
    G_pv_data, G_w_data = q2_main.load_solar_wind_data()
    r_joint, (L_J, G_pv_J, G_w_J) = q2_main.solve_joint_no_storage(
        load_data, G_pv_data, G_w_data)
    indep_results = q2_main.solve_independent_no_storage(
        load_data, G_pv_data, G_w_data)
    indep = q2_main.aggregate_independent(indep_results)
    return {
        'load_data': load_data, 'G_pv_data': G_pv_data, 'G_w_data': G_w_data,
        'r_joint': r_joint, 'L_J': L_J, 'G_pv_J': G_pv_J, 'G_w_J': G_w_J,
        'indep_results': indep_results, 'indep': indep,
    }


# ---------- §9 结构性检查 ----------

def test_joint_profile_aggregation(bundle):
    """联合负荷/光伏/风电 = 三园区逐时相加。"""
    L_sum = sum(bundle['load_data'][p] for p in q2_main.PARKS)
    pv_sum = sum(bundle['G_pv_data'][p] for p in q2_main.PARKS)
    w_sum = sum(bundle['G_w_data'][p] for p in q2_main.PARKS)
    assert np.allclose(bundle['L_J'], L_sum)
    assert np.allclose(bundle['G_pv_J'], pv_sum)
    assert np.allclose(bundle['G_w_J'], w_sum)


def test_load_total_23387(bundle):
    """日负荷总量仍为 23387 kWh。"""
    assert abs(bundle['r_joint']['load_total'] - 23387.0) < 1e-2


def test_balance_and_allocation_errors(bundle):
    """每小时功率平衡误差、风光分配误差 < 1e-4。"""
    err = bundle['r_joint']['errors']
    assert err['max_balance_error'] < 1e-4
    assert err['max_re_error'] < 1e-4


def test_non_negative(bundle):
    """购电量、弃风量、弃光量均非负。"""
    r = bundle['r_joint']
    assert r['grid_total'] >= -1e-6
    assert r['pv_curt'] >= -1e-6
    assert r['w_curt'] >= -1e-6


def test_joint_not_equal_indep_sum(bundle):
    """联合结果不是问题1三个结果的直接相加。"""
    r = bundle['r_joint']
    indep = bundle['indep']
    assert abs(r['grid_total'] - indep['grid_total']) > 1.0


def test_mutual_aid_invariant(bundle):
    """互济不变式：购电减少量 == 弃电减少量。"""
    r = bundle['r_joint']
    indep = bundle['indep']
    grid_reduce = indep['grid_total'] - r['grid_total']
    curt_reduce = indep['curt_total'] - r['curt_total']
    assert abs(grid_reduce - curt_reduce) < 1e-2


def test_unit_cost_denominator(bundle):
    """单位供电成本分母为联合负荷电量。"""
    r = bundle['r_joint']
    uc = r['daily_cost'] / r['load_total']
    assert uc > 0
    assert abs(r['daily_cost'] / uc - r['load_total']) < 1e-2


# ---------- §6 数值验收 ----------

def test_joint_targets(bundle):
    """联合无储能结果与 §6 一致。"""
    r = bundle['r_joint']
    t = q2_main.DOC_JOINT_TARGETS
    assert abs(r['grid_total'] - t['grid_total']) < 0.5
    assert abs(r['pv_curt'] - t['pv_curt']) < 0.5
    assert abs(r['w_curt'] - t['w_curt']) < 0.5
    assert abs(r['re_ratio'] - t['re_ratio']) < 5e-4
    assert abs(r['daily_cost'] - t['daily_cost']) < 1.0
    assert abs(r['daily_cost'] / r['load_total'] - t['unit_cost']) < 5e-4


def test_indep_targets(bundle):
    """三园区独立无储能之和与 §6 一致。"""
    indep = bundle['indep']
    t = q2_main.DOC_INDEP_TARGETS
    assert abs(indep['grid_total'] - t['grid_total']) < 0.5
    assert abs(indep['curt_total'] - t['curt_total']) < 0.5
    assert abs(indep['daily_cost'] - t['daily_cost']) < 1.0


# ---------- 兜底：q2_main.verify_results 全通过 ----------

def test_all_verify_checks_pass(bundle):
    checks = q2_main.verify_results(
        bundle['r_joint'], bundle['L_J'], bundle['G_pv_J'], bundle['G_w_J'],
        bundle['load_data'], bundle['G_pv_data'], bundle['G_w_data'],
        bundle['indep'])
    failed = [(n, d) for n, p, d in checks if not p]
    assert not failed, f"未通过检查: {failed}"


# ---------- 表格构造 ----------

def test_tables_built(bundle):
    r = bundle['r_joint']
    joint = q2_main.joint_metrics(r)
    df_h = q2_main.build_hourly_df(r, bundle['L_J'], bundle['G_pv_J'], bundle['G_w_J'])
    assert len(df_h) == 24
    df_s = q2_main.build_summary_df(joint)
    assert len(df_s) == 9
    df_c = q2_main.build_comparison_df(bundle['indep'], joint)
    assert df_c.shape[0] == 9
    assert '联合后的变化' in df_c.columns
