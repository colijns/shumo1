# -*- coding: utf-8 -*-
"""
问题1 主程序
功能：统一 MILP 模型求解三个园区、三个场景的所有结果
     - 无储能
     - 固定 50 kW/100 kWh 储能
     - 优化容量（连续理论最优 + 工程整数最优 + 局部网格验证）
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
    run_no_storage,
    run_fixed_storage, run_optimized_storage,
    run_engineering_storage,
)
from grid_search import grid_search_capacity

# 工程容量粒度与搜索上界
P_STEP = 5   # kW
E_STEP = 10  # kWh
P_MAX = 200  # kW
E_MAX = 600  # kWh


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
    all_soc = []  # SOC包含 t=0...24 共25个状态点
    all_summary = []  # 方案汇总
    final_results = {}
    local_grids = {}

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
        for t in range(25):
            all_soc.append({
                '园区': park,
                '场景': '50kW/100kWh',
                'SOC时点': t,
                '储能电量(kWh)': r1['E'][t],
                'SOC': r1['E'][t] / 100,
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

        # ----- 2d. 工程整数最优 -----
        print(f'\n  [2d] 工程整数优化...')
        print(f'     工程粒度: {P_STEP} kW / {E_STEP} kWh')
        r3 = run_engineering_storage(
            L, P, W,
            P_step=P_STEP, E_step=E_STEP,
            P_max=P_MAX, E_max=E_MAX,
        )
        if r3['success']:
            P_engineering = r3['P_ess']
            E_engineering = r3['E_ess']
            print(f'     理论连续最优: {r2["P_ess"]:.1f} kW / {r2["E_ess"]:.1f} kWh')
            print(f'     工程整数最优: {P_engineering:.0f} kW / {E_engineering:.0f} kWh')
            print(f'     工程方案日运行成本: {r3["daily_cost"]:.2f} 元')
            print(f'     工程方案年综合成本: {r3["annual_cost"]:.2f} 元')

            for t in range(24):
                soc = r3['E'][t] / E_engineering if E_engineering > 0 else 0
                all_hourly.append({
                    '园区': park,
                    '场景': f'工程最优({P_engineering:.0f}kW/{E_engineering:.0f}kWh)',
                    '时刻': t,
                    '负荷': L[t], '光伏出力': P[t], '风电出力': W[t],
                    '光伏直供': r3['P_load_pv'][t], '风电直供': r3['P_load_w'][t],
                    '充电功率': r3['P_ch'][t], '放电功率': r3['P_dis'][t],
                    'SOC': soc,
                    '电网购电': r3['P_grid'][t],
                    '弃光': r3['P_curt_pv'][t], '弃风': r3['P_curt_w'][t],
                })
            if E_engineering > 0:
                for t in range(25):
                    all_soc.append({
                        '园区': park,
                        '场景': f'工程最优({P_engineering:.0f}kW/{E_engineering:.0f}kWh)',
                        'SOC时点': t,
                        '储能电量(kWh)': r3['E'][t],
                        'SOC': r3['E'][t] / E_engineering,
                    })
            final_results[park] = {
                'continuous': r2,
                'engineering': r3,
            }
        else:
            print(f'     ⚠ 工程整数优化失败: {r3["status"]}')
            r3 = None

        # ----- 2e. 统一粒度的局部网格验证 -----
        print(f'\n  [2e] 局部网格验证...')
        p_center = int(round(r3['P_ess'])) if r3 else 0
        e_center = int(round(r3['E_ess'])) if r3 else 0
        gs = grid_search_capacity(
            L, P, W,
            P_range=(max(0, p_center - 20), min(P_MAX, p_center + 20), P_STEP),
            E_range=(max(0, e_center - 40), min(E_MAX, e_center + 40), E_STEP),
            verbose=False
        )
        if gs['best']['annual_cost'] < float('inf'):
            gs_best = gs['best']
            local_grids[park] = gs
            print(f'     局部网格最低点: P={gs_best["P_ess"]:.0f} kW, '
                  f'E={gs_best["E_ess"]:.0f} kWh')
            print(f'     年成本: {gs_best["annual_cost"]:.2f} 元')
            matches_engineering = (
                r3 is not None
                and abs(gs_best['P_ess'] - r3['P_ess']) < 1e-6
                and abs(gs_best['E_ess'] - r3['E_ess']) < 1e-6
            )
            print(f'     与工程整数最优一致: {"是" if matches_engineering else "否"}')
        else:
            gs_best = None
            print(f'     ⚠ 局部网格无有效结果')

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

        # 工程整数最优容量
        if r3 and r3['success']:
            _add_summary(all_summary, park,
                         f'工程最优({r3["P_ess"]:.0f}kW/{r3["E_ess"]:.0f}kWh)',
                         r3['P_ess'], r3['E_ess'],
                         r3['load_total'], r3['grid_total'],
                         r3['pv_curt'], r3['w_curt'],
                         r3['re_ratio'], r3['daily_cost'],
                         r3['annual_cost'], r3['unit_annual_cost'])

        # 结果检查
        print(f'\n  [检查] 结果合理性...')
        _check_results(r0, r1, r3, park)

    # ===== 3. 输出 =====
    print(f'\n{"=" * 50}')
    print('数据导出')

    # 逐时调度表与25点SOC轨迹
    df_hourly = pd.DataFrame(all_hourly)
    df_soc = pd.DataFrame(all_soc)
    hourly_path = os.path.join(output_dir, '问题1_逐时调度表.xlsx')
    with pd.ExcelWriter(hourly_path, engine='openpyxl') as writer:
        df_hourly.to_excel(writer, sheet_name='逐时调度', index=False)
        df_soc.to_excel(writer, sheet_name='SOC轨迹', index=False)
    _format_excel(hourly_path)
    print(f'  逐时调度表 -> {hourly_path}')

    # 方案汇总表
    df_summary = pd.DataFrame(all_summary)
    df_summary = _add_savings_columns(df_summary)
    summary_path = os.path.join(output_dir, '问题1_方案汇总表.xlsx')
    df_summary.to_excel(summary_path, index=False)
    _format_excel(summary_path)
    print(f'  方案汇总表 -> {summary_path}')
    print(f'\n{df_summary.to_string(index=False)}')

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
        row['年运行成本(元)'] = round(365 * daily_cost, 2)
        row['储能年均投资(元)'] = round((C_P_ESS * P_ess + C_E_ESS * E_ess) / Y, 2)
        row['年综合成本(元)'] = round(annual_cost, 2)
        row['单位供电成本(元/kWh)'] = round(unit_cost, 4)
        row['求解状态'] = 'Optimal'
    summary.append(row)


def _add_savings_columns(df):
    """增加相对无储能和固定50/100方案的经济性指标。"""
    if df.empty:
        return df

    df = df.copy()
    df['相对无储能年节省(元)'] = 0.0
    df['相对50/100年节省(元)'] = 0.0
    for park in df['园区'].unique():
        mask = df['园区'] == park
        park_df = df.loc[mask]
        no_storage = park_df.loc[park_df['方案'] == '无储能', '年综合成本(元)']
        fixed = park_df.loc[park_df['方案'] == '50kW/100kWh', '年综合成本(元)']
        if not no_storage.empty:
            df.loc[mask, '相对无储能年节省(元)'] = (
                no_storage.iloc[0] - df.loc[mask, '年综合成本(元)']
            ).round(2)
        if not fixed.empty:
            df.loc[mask, '相对50/100年节省(元)'] = (
                fixed.iloc[0] - df.loc[mask, '年综合成本(元)']
            ).round(2)
    return df


def _format_excel(path):
    """设置列宽、冻结标题行并启用筛选。"""
    import unicodedata

    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = 'A2'
        worksheet.auto_filter.ref = worksheet.dimensions
        worksheet.sheet_view.showGridLines = False

        for cell in worksheet[1]:
            cell.fill = PatternFill('solid', fgColor='1F4E78')
            cell.font = Font(color='FFFFFF', bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        worksheet.row_dimensions[1].height = 32

        for column in worksheet.columns:
            values = [str(cell.value) if cell.value is not None else '' for cell in column]
            display_lengths = [
                sum(2 if unicodedata.east_asian_width(char) in 'WFA' else 1 for char in value)
                for value in values
            ]
            width = min(max(max(display_lengths) + 2, 12), 30)
            worksheet.column_dimensions[column[0].column_letter].width = width

        headers = {cell.column: str(cell.value) for cell in worksheet[1]}
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                header = headers.get(cell.column, '')
                if header == 'SOC':
                    cell.number_format = '0.0000'
                elif any(key in header for key in ['成本', '投资', '节省']):
                    cell.number_format = '#,##0.00'
                elif isinstance(cell.value, float):
                    cell.number_format = '#,##0.000'
    workbook.save(path)


def _check_results(r_no_sto, r_fixed, r_final, park_name):
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

    # 固定储能与最终工程方案都执行完整检查
    for label, result in [('固定储能', r_fixed), ('工程最优', r_final)]:
        if not result or not result['success']:
            continue
        err = result['errors']
        checks.append((f'{label}功率平衡误差', err['max_balance_error'] < 1e-4))
        checks.append((f'{label}风光分配误差', err['max_re_error'] < 1e-4))
        checks.append((f'{label}SOC下限合规', err['min_soc'] >= S_MIN - 1e-6))
        checks.append((f'{label}SOC上限合规', err['max_soc'] <= S_MAX + 1e-6))
        checks.append((f'{label}日末SOC回归', err['end_soc_error'] < 1e-6))
        checks.append((f'{label}能量守恒', err['net_energy_error'] < 1e-6))

        # 储能降低购电量
        if r_no_sto['success'] and result['E_ess'] > 0:
            grid_reduction = r_no_sto['grid_total'] - result['grid_total']
            checks.append((f'{label}降低购电', grid_reduction >= -1e-6))

    for name, passed in checks:
        icon = '✓' if passed else '✗'
        print(f'     {icon} {name}')


if __name__ == '__main__':
    df_hourly, df_summary = run_q1()
    print('\nq1_main demo 运行完毕')
