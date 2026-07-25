# -*- coding: utf-8 -*-
"""
网格搜索验证模块
功能：对储能功率和容量进行网格搜索，验证 MILP 连续优化结果
      生成"容量—成本"关系图数据
"""

import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_loader import C_P_ESS, C_E_ESS, Y
from milp_model import run_fixed_capacity


def grid_search_capacity(load, G_pv, G_w,
                          P_range=(0, 300, 5),
                          E_range=(0, 1000, 10),
                          verbose=True,
                          curt_penalty=0.0,
                          surplus_only_charge=False,
                          min_accom_rate=None):
    """
    网格搜索最优储能容量。

    参数：
        load, G_pv, G_w : 24 时段数据
        P_range : (min, max, step) 功率搜索范围 kW
        E_range : (min, max, step) 容量搜索范围 kWh
        verbose : 是否打印进度
        curt_penalty : 弃电惩罚因子 λ (元/kWh)，透传给 solve_park_optimization。
                       默认 0 = View A，与原行为一致。
        surplus_only_charge : True 时储能只能吸收当小时风光余电。
        min_accom_rate : 最低风光消纳率 R0（ε-约束法）。None 时不加约束；给定 R0 时，
                         不满足消纳率的 (P,E) 点不可行（success=False）被自动跳过，
                         剩余可行点中取年成本最低者，即工程整数 ε-约束最优。

    返回：
        dict:
            'best': {'P_ess', 'E_ess', 'annual_cost', 'daily_cost', 're_ratio', 'curt_total'}
            'grid': list of all evaluated feasible points (含 re_ratio, curt_total)
    """
    P_min, P_max, P_step = P_range
    E_min, E_max, E_step = E_range

    P_vals = np.arange(P_min, P_max + P_step, P_step)
    E_vals = np.arange(E_min, E_max + E_step, E_step)

    best_cost = float('inf')
    best_P = 0
    best_E = 0
    best_daily = 0
    best_re_ratio = 0.0
    best_curt = 0.0
    grid_points = []

    total = len(P_vals) * len(E_vals)
    count = 0

    for p in P_vals:
        for e in E_vals:
            count += 1
            if verbose and count % 500 == 0:
                print(f'  网格搜索进度: {count}/{total} ({100*count/total:.0f}%)')

            result = run_fixed_capacity(load, G_pv, G_w, P_ess=p, E_ess=e,
                                        curt_penalty=curt_penalty,
                                        surplus_only_charge=surplus_only_charge,
                                        min_accom_rate=min_accom_rate)
            if not result['success']:
                continue

            annual_cost = 365 * result['daily_cost'] + (C_P_ESS * p + C_E_ESS * e) / Y

            grid_points.append({
                'P_ess': p,
                'E_ess': e,
                'daily_cost': result['daily_cost'],
                'annual_cost': annual_cost,
                're_ratio': result['re_ratio'],
                'curt_total': result['curt_total'],
            })

            if annual_cost < best_cost:
                best_cost = annual_cost
                best_P = p
                best_E = e
                best_daily = result['daily_cost']
                best_re_ratio = result['re_ratio']
                best_curt = result['curt_total']

    if verbose:
        print(f'网格搜索完成: 共评估 {len(grid_points)} 个点')
        print(f'最优: P={best_P} kW, E={best_E} kWh, 年成本={best_cost:.2f} 元')

    return {
        'best': {
            'P_ess': best_P,
            'E_ess': best_E,
            'annual_cost': best_cost,
            'daily_cost': best_daily,
            're_ratio': best_re_ratio,
            'curt_total': best_curt,
        },
        'grid': grid_points,
    }


if __name__ == '__main__':
    from data_loader import load_load_data, load_solar_wind_data

    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    park = 'A'
    print(f'\n===== 园区 {park} 网格搜索 =====')
    gs = grid_search_capacity(
        load_data[park], G_pv_data[park], G_w_data[park],
        P_range=(0, 200, 10),
        E_range=(0, 600, 20),
        verbose=True
    )
    print(f'最优: P={gs["best"]["P_ess"]} kW, '
          f'E={gs["best"]["E_ess"]} kWh, '
          f'年成本={gs["best"]["annual_cost"]:.2f} 元')
    print('grid_search demo 运行完毕')
