# -*- coding: utf-8 -*-
"""
问题1 主程序
功能：统一 MILP 模型求解三个园区、三个场景的所有结果
     - 无储能
     - 固定 50 kW/100 kWh 储能
     - 优化容量（连续优化 + 工程取整 + 网格搜索验证）
"""

import sys
import os
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))  # 确保 Q1 目录在路径中
from data_loader import (
    load_load_data, load_solar_wind_data, PARKS,
    C_PV, C_W, C_G, C_P_ESS, C_E_ESS, Y,
    ETA_C, ETA_D, S_MIN, S_MAX, S_0, DT,
)
from milp_model import (
    solve_park_optimization, run_no_storage,
    run_fixed_storage, run_optimized_storage,
    run_fixed_capacity,
)
from grid_search import grid_search_capacity
from visualization import (
    plot_load_and_re_gen, plot_storage_dispatch,
    plot_comparison, plot_cost_heatmap,
    plot_grid_search_cost_curve,
)

# 工程取整步长
P_STEP = 5   # kW
E_STEP = 10  # kWh


def round_to_step(val, step):
    """向最近的 step 取整（四舍五入，最少为 0）"""
    return max(0, round(val / step) * step)


def run_q1(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    """问题1完整求解流程"""
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. 加载数据 =====
    print('=' * 60)
    print('问题1：各园区独立运营储能配置方案及经济性分析')
    print('=' * 60)

    print('\n[1/5] 读取数据...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    parks = ['A', 'B', 'C']
    all_hourly = []  # 逐时结果汇总
    all_summary = []  # 方案汇总

    # ===== 2. 循环处理每个园区 =====
    for park in parks:
        print(f'\n{"=" * 50}')
        print(f'【园区 {park}】')
        print(f'  光伏: {PARKS[park]["pv"]} kW, 风电: {PARKS[park]["w"]} kW')
        print(f'  {max(load_data[park]):.0f} kW')
        print(f'{"=" * 50}')

        L = load_data[park]
        P = G_pv_data[park]
        W = G_w_data[park]

        # ----- 2a. 无储能 -----
        print(f'\n  [2a] 无储能方案求解...')
        t0 = time.time()
        r0 = run_no_storage(L, P, W)
        t1 = time.time()
        if not r0['success']:
            print(f'  ❌ 无储能求解失败: {r0["status"]}')
            continue
        print(f'     ✓ 求解完成 ({t1-t0:.2f}s)')
        print(f'     日运行成本: {r0["daily_cost"]:.2f} 元')
        print(f'     购电量: {r0["grid_total"]:.2f} kWh')
        print(f'     弃电量: {r0["curt_total"]:.2f} kWh')
        print(f'     风光消纳率: {r0["re_ratio"]*100:.1f}%')

        # 保存逐时结果
        for t in range(24):
            all_hourly.append({
                '园区': park, '场景': '无储能', '时刻': t,
                '负荷': L[t], '光伏出力': P[t], '风电出力': W[t],
                '光伏直供': r0['P_load_pv'][t], '风电直供': r0['P_load_w'][t],
                '充电功率': 0, '放电功率': 0, 'SOC': 0,
                '电网购电': r0['P_grid'][t],
                '弃光': r0['P_curt_pv'][t], '弃风': r0['P_curt_w'][t],
            })

        # ----- 2b. 固定 50 kW/100 kWh 储能 -----
        print(f'\n  [2b] 固定 50kW/100kWh 储能求解...')
        t0 = time.time()
        r1 = run_fixed_storage(L, P, W, 50, 100)
        t1 = time.time()
        if not r1['success']:
            print(f'  ❌ 固定储能求解失败: {r1["status"]}')
            continue
        print(f'     ✓ 求解完成 ({t1-t0:.2f}s)')
        print(f'     日运行成本: {r1["daily_cost"]:.2f} 元')
        print(f'     年综合成本: {r1["annual_cost"]:.2f} 元')
        print(f'     年均投资: {r1["inv_annual"]:.2f} 元')
        print(f'     购电量: {r1["grid_total"]:.2f} kWh')
        print(f'     弃电量: {r1["curt_total"]:.2f} kWh')
        print(f'     风光消纳率: {r1["re_ratio"]*100:.1f}%')

        for t in range(24):
            soc = r1['E'][t] / 100 if 100 > 0 else 0
            all_hourly.append({
                '园区': park, '场景': '50kW/100kWh', '时刻': t,
                '负荷': L[t], '光伏出力': P[t], '风电出力': W[t],
                '光伏直供': r1['P_load_pv'][t], '风电直供': r1['P_load_w'][t],
                '充电功率': r1['P_ch'][t], '放电功率': r1['P_dis'][t],
                'SOC': soc,
                '电网购电': r1['P_grid'][t],
                '弃光': r1['P_curt_pv'][t], '弃风': r1['P_curt_w'][t],
            })

        # ----- 2c. 优化容量（连续变量）-----
        print(f'\n  [2c] 容量优化求解...')
        t0 = time.time()
        r2 = run_optimized_storage(L, P, W)
        t1 = time.time()
        if not r2['success']:
            print(f'  ❌ 容量优化求解失败: {r2["status"]}')
            continue
        print(f'     ✓ 求解完成 ({t1-t0:.2f}s)')
        print(f'     最优功率: {r2["P_ess"]:.1f} kW')
        print(f'     最优容量: {r2["E_ess"]:.1f} kWh')
        print(f'     年综合成本: {r2["annual_cost"]:.2f} 元')
        print(f'     日运行成本: {r2["daily_cost"]:.2f} 元')

        # ----- 2d. 工程取整 -----
        print(f'\n  [2d] 工程取整...')
        P_rounded = round_to_step(r2['P_ess'], P_STEP)
        E_rounded = round_to_step(r2['E_ess'], E_STEP)
        print(f'     理论: {r2["P_ess"]:.1f} kW / {r2["E_ess"]:.1f} kWh')
        print(f'     取整: {P_rounded:.0f} kW / {E_rounded:.0f} kWh')

        r3 = solve_park_optimization(L, P, W,
                                      P_ess=P_rounded, E_ess=E_rounded,
                                      optimize_capacity=False)
        if r3['success']:
            print(f'     取整后日运行成本: {r3["daily_cost"]:.2f} 元')
            print(f'     取整后年综合成本: {r3["annual_cost"]:.2f} 元')

            for t in range(24):
                soc = r3['E'][t] / E_rounded if E_rounded > 0 else 0
                all_hourly.append({
                    '园区': park, '场景': f'优化({P_rounded:.0f}kW/{E_rounded:.0f}kWh)', '时刻': t,
                    '负荷': L[t], '光伏出力': P[t], '风电出力': W[t],
                    '光伏直供': r3['P_load_pv'][t], '风电直供': r3['P_load_w'][t],
                    '充电功率': r3['P_ch'][t], '放电功率': r3['P_dis'][t],
                    'SOC': soc,
                    '电网购电': r3['P_grid'][t],
                    '弃光': r3['P_curt_pv'][t], '弃风': r3['P_curt_w'][t],
                })
        else:
            r3 = None
            print(f'     ⚠ 取整方案求解失败')

        # ----- 2e. 网格搜索验证 -----
        print(f'\n  [2e] 网格搜索验证（稍候...）')
        gs = grid_search_capacity(
            L, P, W,
            P_range=(0, 200, 10),
            E_range=(0, 600, 20),
            verbose=False
        )
        if gs['best']['annual_cost'] < float('inf'):
            gs_best = gs['best']
            print(f'     网格搜索最优: P={gs_best["P_ess"]:.0f} kW, '
                  f'E={gs_best["E_ess"]:.0f} kWh')
            print(f'     年成本: {gs_best["annual_cost"]:.2f} 元')
        else:
            gs_best = None
            print(f'     ⚠ 网格搜索无有效结果')

        # ----- 2f. 汇总结果 -----
        print(f'\n  [汇总] 园区 {park} 各方案对比')

        # 无储能
        no_sto_cost_annual = 365 * r0['daily_cost']
        _add_summary(all_summary, park, '无储能', 0, 0,
                     r0['load_total'], r0['grid_total'],
                     r0['pv_curt'], r0['w_curt'],
                     r0['re_ratio'], r0['daily_cost'],
                     no_sto_cost_annual, r0['unit_daily_cost'])

        # 固定储能
        _add_summary(all_summary, park, '50kW/100kWh', 50, 100,
                     r1['load_total'], r1['grid_total'],
                     r1['pv_curt'], r1['w_curt'],
                     r1['re_ratio'], r1['daily_cost'],
                     r1['annual_cost'], r1['unit_annual_cost'])

        # 优化容量（取整）
        if r3 and r3['success']:
            _add_summary(all_summary, park,
                         f'优化({P_rounded:.0f}kW/{E_rounded:.0f}kWh)',
                         P_rounded, E_rounded,
                         r3['load_total'], r3['grid_total'],
                         r3['pv_curt'], r3['w_curt'],
                         r3['re_ratio'], r3['daily_cost'],
                         r3['annual_cost'], r3['unit_annual_cost'])

        # 网格搜索最优
        if gs_best and gs_best['P_ess'] > 0 and gs_best['E_ess'] > 0:
            # 用最优容量再跑一次得到详细指标
            gs_detail = run_fixed_capacity(L, P, W,
                                           gs_best['P_ess'], gs_best['E_ess'])
            if gs_detail['success']:
                _add_summary(all_summary, park,
                             f'网格最优({gs_best["P_ess"]:.0f}kW/{gs_best["E_ess"]:.0f}kWh)',
                             gs_best['P_ess'], gs_best['E_ess'],
                             gs_detail['load_total'], gs_detail['grid_total'],
                             gs_detail['pv_curt'], gs_detail['w_curt'],
                             gs_detail['re_ratio'], gs_detail['daily_cost'],
                             gs_detail['annual_cost'], gs_detail['unit_annual_cost'])

        # 结果检查
        print(f'\n  [检查] 结果合理性...')
        _check_results(r0, r1, park)

    # ===== 3. 输出 =====
    print(f'\n{"=" * 50}')
    print('数据导出')

    # 逐时调度表
    df_hourly = pd.DataFrame(all_hourly)
    hourly_path = os.path.join(output_dir, '问题1_逐时调度表.xlsx')
    df_hourly.to_excel(hourly_path, index=False)
    print(f'  逐时调度表 -> {hourly_path}')

    # 方案汇总表
    df_summary = pd.DataFrame(all_summary)
    summary_path = os.path.join(output_dir, '问题1_方案汇总表.xlsx')
    df_summary.to_excel(summary_path, index=False)
    print(f'  方案汇总表 -> {summary_path}')
    print(f'\n{df_summary.to_string(index=False)}')

    # ===== 4. 可视化 =====
    print(f'\n{"=" * 50}')
    print('生成图表...')

    for park in parks:
        L = load_data[park]
        P = G_pv_data[park]
        W = G_w_data[park]

        # 负荷与风光出力
        plot_load_and_re_gen(L, P, W, park,
                             save_path=os.path.join(output_dir, f'园区{park}_负荷与风光出力.png'))

        # 固定储能调度图
        r1 = run_fixed_storage(L, P, W, 50, 100)
        if r1['success']:
            plot_storage_dispatch(r1, park, '50kW/100kWh', 50, 100,
                                  save_path=os.path.join(output_dir, f'园区{park}_固定储能调度.png'))

        # 优化容量调度图
        P_round = round_to_step(r2['P_ess'], P_STEP) if 'r2' in dir() and r2.get('success') else 50
        E_round = round_to_step(r2['E_ess'], E_STEP) if 'r2' in dir() and r2.get('success') else 100

        # 方案对比图
        plot_comparison(df_summary, park,
                        save_path=os.path.join(output_dir, f'园区{park}_方案对比.png'))

    # 网格搜索热力图（园区 B/C 有经济性时出图）
    if gs_best and gs_best['P_ess'] > 0:
        for park in ['B', 'C']:
            L = load_data[park]
            P = G_pv_data[park]
            W = G_w_data[park]
            gs = grid_search_capacity(L, P, W,
                                       P_range=(0, 150, 10),
                                       E_range=(0, 400, 20),
                                       verbose=False)
            plot_cost_heatmap(gs['grid'], park,
                              save_path=os.path.join(output_dir, f'园区{park}_容量成本热力图.png'))
            plot_grid_search_cost_curve(gs['grid'], park,
                                        save_path=os.path.join(output_dir, f'园区{park}_容量成本曲线.png'))

    print(f'\n{"=" * 50}')
    print('问题1 求解完毕!')
    print(f'{"=" * 50}')

    return df_hourly, df_summary


def _add_summary(summary, park, plan, P_ess, E_ess,
                 load_total, grid_total,
                 pv_curt, w_curt, re_ratio,
                 daily_cost, annual_cost, unit_cost):
    """添加方案汇总行"""
    row = {
        '园区': park,
        '方案': plan,
        '储能功率(kW)': P_ess,
        '储能容量(kWh)': E_ess,
        '负荷电量(kWh)': round(load_total, 1) if load_total else None,
    }
    if grid_total is not None:
        row['电网购电量(kWh)'] = round(grid_total, 1)
        row['弃光量(kWh)'] = round(pv_curt, 1)
        row['弃风量(kWh)'] = round(w_curt, 1)
        row['风光消纳率(%)'] = round(re_ratio * 100, 1)
        row['日运行成本(元)'] = round(daily_cost, 2)
        row['年综合成本(元)'] = round(annual_cost, 2)
        row['单位供电成本(元/kWh)'] = round(unit_cost, 4)
    summary.append(row)


def _check_results(r_no_sto, r_fixed, park_name):
    """检查结果合理性"""
    checks = []

    # 无储能检查
    if r_no_sto['success']:
        # 购电量非负
        checks.append(('购电量非负', r_no_sto['grid_total'] >= -1e-6))
        # 弃电量非负
        checks.append(('弃电量非负', r_no_sto['curt_total'] >= -1e-6))
        # 消纳率在[0,1]
        checks.append(('消纳率合理', 0 <= r_no_sto['re_ratio'] <= 1))

    # 固定储能检查
    if r_fixed['success']:
        err = r_fixed['errors']
        checks.append(('功率平衡误差', err['max_balance_error'] < 1e-4))
        checks.append(('风光分配误差', err['max_re_error'] < 1e-4))
        checks.append(('SOC下限合规', err['min_soc'] >= S_MIN - 1e-6))
        checks.append(('SOC上限合规', err['max_soc'] <= S_MAX + 1e-6))
        checks.append(('日末SOC回归', err['end_soc_error'] < 1e-6))
        checks.append(('能量守恒', err['net_energy_error'] < 1e-6))

        # 储能降低购电量
        if r_no_sto['success']:
            grid_reduction = r_no_sto['grid_total'] - r_fixed['grid_total']
            checks.append(('储能降低购电', grid_reduction >= -1e-6))

    for name, passed in checks:
        icon = '✓' if passed else '✗'
        print(f'     {icon} {name}')


if __name__ == '__main__':
    df_hourly, df_summary = run_q1()
    print('\nq1_main demo 运行完毕')
