# -*- coding: utf-8 -*-
"""
问题1 可视化出图脚本 — 读取 xlsx 结果并生成 PNG 图表

用法：conda activate math && python Q1/plot_results.py

依赖 q1_main.py 运行完毕，输出目录 Q1/output/ 中存在：
  - 问题1_方案汇总表.xlsx
  - 问题1_逐时调度表.xlsx
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_loader import load_load_data, load_solar_wind_data
from milp_model import run_fixed_storage, run_optimized_storage, run_engineering_storage
from grid_search import grid_search_capacity
from visualization import (
    plot_load_and_re_gen, plot_storage_dispatch,
    plot_comparison, plot_cost_heatmap,
    plot_grid_search_cost_curve,
)

# 与 q1_main 一致的工程粒度
P_STEP = 5
E_STEP = 10
P_MAX = 200
E_MAX = 600

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
SUMMARY_PATH = os.path.join(OUTPUT_DIR, '问题1_方案汇总表.xlsx')
HOURLY_PATH = os.path.join(OUTPUT_DIR, '问题1_逐时调度表.xlsx')
PARKS = ['A', 'B', 'C']


def plot_all():
    if not os.path.exists(SUMMARY_PATH):
        print(f'[错误] 未找到 {SUMMARY_PATH}，请先运行 python Q1/q1_main.py')
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('读取数据...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()
    df_summary = pd.read_excel(SUMMARY_PATH)

    for park in PARKS:
        print(f'\n===== 园区 {park} =====')
        L = load_data[park]
        P = G_pv_data[park]
        W = G_w_data[park]

        # —— 负荷与风光出力 ——
        print('  负荷与风光出力曲线...')
        plot_load_and_re_gen(
            L, P, W, park,
            save_path=os.path.join(OUTPUT_DIR, f'园区{park}_负荷与风光出力.png'),
        )

        # —— 固定储能调度图 ——
        print('  固定储能调度图...')
        r_fixed = run_fixed_storage(L, P, W)
        if r_fixed['success']:
            plot_storage_dispatch(
                r_fixed, park, '50kW/100kWh', 50, 100,
                save_path=os.path.join(OUTPUT_DIR, f'园区{park}_固定储能调度.png'),
            )

        # —— 工程最优调度图 ——
        print('  工程最优调度图...')
        r_eng = run_engineering_storage(L, P, W, P_STEP, E_STEP, P_MAX, E_MAX)
        if r_eng['success'] and r_eng['E_ess'] > 0:
            plot_storage_dispatch(
                r_eng, park,
                f'工程最优{r_eng["P_ess"]:.0f}kW/{r_eng["E_ess"]:.0f}kWh',
                r_eng['P_ess'], r_eng['E_ess'],
                save_path=os.path.join(OUTPUT_DIR, f'园区{park}_工程最优调度.png'),
            )

        # —— 方案对比图 ——
        print('  方案对比图...')
        plot_comparison(
            df_summary, park,
            save_path=os.path.join(OUTPUT_DIR, f'园区{park}_方案对比.png'),
        )

    # —— 容量成本热力图 & 曲线（园区 B/C 有储能经济性）——
    for park in ['B', 'C']:
        print(f'\n===== 园区 {park} 成本曲面 =====')
        L = load_data[park]
        P = G_pv_data[park]
        W = G_w_data[park]
        gs = grid_search_capacity(
            L, P, W,
            P_range=(0, min(150, P_MAX), P_STEP * 2),
            E_range=(0, min(400, E_MAX), E_STEP * 2),
            verbose=False,
        )
        if gs['best']['annual_cost'] < float('inf'):
            print('  容量成本热力图...')
            plot_cost_heatmap(
                gs['grid'], park,
                save_path=os.path.join(OUTPUT_DIR, f'园区{park}_容量成本热力图.png'),
            )
            print('  容量成本曲线...')
            plot_grid_search_cost_curve(
                gs['grid'], park,
                save_path=os.path.join(OUTPUT_DIR, f'园区{park}_容量成本曲线.png'),
            )

    print(f'\n全部图表已保存至 {OUTPUT_DIR}')


if __name__ == '__main__':
    plot_all()
    print('plot_results demo 运行完毕')
