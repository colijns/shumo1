# -*- coding: utf-8 -*-
"""
可视化模块 —— 问题1 结果出图
功能：
  1. 负荷与风光出力曲线
  2. 储能充放电功率与 SOC 曲线
  3. 各方案购电量、弃电量和成本对比图
  4. 容量-成本热力图（网格搜索）
"""

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'templates'))
from common.plot_style import set_chinese_style, save_fig


def _fix_chinese_font():
    """在 set_chinese_style 之后确保中文字体生效"""
    for _fp in [
        r'C:\Windows\Fonts\simhei.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
    ]:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            _name = font_manager.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


def plot_load_and_re_gen(load, G_pv, G_w, park_name, save_path=None):
    """绘制负荷与风光出力曲线"""
    set_chinese_style()
    _fix_chinese_font()
    t = np.arange(24)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, load, 'k-', linewidth=2, label='负荷')
    ax.fill_between(t, 0, G_pv, alpha=0.3, color='orange', label='光伏出力')
    ax.fill_between(t, 0, G_w, alpha=0.3, color='steelblue', label='风电出力')
    ax.set_xlabel('时刻 (h)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title(f'园区 {park_name} 典型日负荷与风光出力')
    ax.set_xticks(t)
    ax.set_xlim(0, 23)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_storage_dispatch(result, park_name, plan_name, P_ess, E_ess, save_path=None):
    """绘制储能充放电功率与SOC曲线"""
    set_chinese_style()
    _fix_chinese_font()
    t = np.arange(24)
    t_soc = np.arange(25)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 上：充放电功率
    ax = axes[0]
    p_ch = np.array(result['P_ch'])
    p_dis = np.array(result['P_dis'])
    ax.bar(t, p_ch, width=0.8, color='green', alpha=0.7, label='充电')
    ax.bar(t, -p_dis, width=0.8, color='red', alpha=0.7, label='放电')
    ax.axhline(P_ess, color='gray', linestyle='--', linewidth=1, label=f'额定功率 {P_ess} kW')
    ax.axhline(-P_ess, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('功率 (kW)')
    ax.set_title(f'园区{park_name} {plan_name} — 储能充放电功率')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 下：SOC
    ax = axes[1]
    soc = np.array(result['E'][:25]) / E_ess if E_ess > 0 else np.zeros(25)
    ax.plot(t_soc, soc * 100, 'b-', linewidth=2, marker='o', markersize=4)
    ax.axhline(90, color='r', linestyle='--', linewidth=1, alpha=0.5, label='SOC上限 (90%)')
    ax.axhline(10, color='r', linestyle='--', linewidth=1, alpha=0.5, label='SOC下限 (10%)')
    ax.set_xlabel('SOC时点（t=24为日末状态）')
    ax.set_ylabel('SOC (%)')
    ax.set_title(f'园区{park_name} {plan_name} — 荷电状态')
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_comparison(summary_df, park_name, save_path=None):
    """绘制各方案对比（购电量、弃电量、年综合成本）"""
    set_chinese_style()
    _fix_chinese_font()
    park_data = summary_df[summary_df['园区'] == park_name].copy()
    if park_data.empty:
        return

    # 只保留有完整数据的行
    park_data = park_data[park_data['电网购电量(kWh)'].notna()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    labels = park_data['方案'].values
    x = np.arange(len(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    # 购电量
    ax = axes[0]
    ax.bar(x, park_data['电网购电量(kWh)'].values, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('电网购电量 (kWh)')
    ax.set_title(f'园区 {park_name} 购电量对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 弃电量
    ax = axes[1]
    curt = park_data['弃光量(kWh)'].fillna(0).values + park_data['弃风量(kWh)'].fillna(0).values
    ax.bar(x, curt, color=colors, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('弃电量 (kWh)')
    ax.set_title(f'园区 {park_name} 弃电量对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 年综合成本
    ax = axes[2]
    annual_cost = park_data['年综合成本(元)'].values / 10000
    bars = ax.bar(x, annual_cost, color=colors, width=0.6)
    ax.bar_label(bars, labels=[f'{value:.2f}' for value in annual_cost],
                 padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('年综合成本 (万元)')
    ax.set_title(f'园区 {park_name} 年综合成本对比')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_cost_heatmap(grid_points, park_name, save_path=None):
    """绘制容量-成本热力图"""
    set_chinese_style()
    _fix_chinese_font()

    if not grid_points:
        return

    df = pd.DataFrame(grid_points)
    pivot = df.pivot_table(
        index='E_ess', columns='P_ess',
        values='annual_cost', aggfunc='mean'
    )
    P_vals = pivot.columns.values
    E_vals = pivot.index.values
    cost_mat = pivot.values

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(P_vals, E_vals, cost_mat, shading='auto',
                       cmap='YlOrRd', norm=plt.Normalize(
                           vmin=cost_mat.min(), vmax=cost_mat.max()))
    cbar = plt.colorbar(im, ax=ax, label='年综合成本 (元)')

    # 标记最小值
    min_idx = np.unravel_index(np.argmin(cost_mat), cost_mat.shape)
    min_P = P_vals[min_idx[1]]
    min_E = E_vals[min_idx[0]]
    ax.plot(min_P, min_E, 'b*', markersize=15, markeredgecolor='white',
            markeredgewidth=1.5, label=f'最优: {min_P:.0f}kW/{min_E:.0f}kWh')

    ax.set_xlabel('储能功率 (kW)')
    ax.set_ylabel('储能容量 (kWh)')
    ax.set_title(f'园区 {park_name} 容量-年综合成本热力图')
    ax.legend(loc='upper right')
    ax.set_xlim(P_vals.min(), P_vals.max())
    ax.set_ylim(E_vals.min(), E_vals.max())

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_grid_search_cost_curve(grid_points, park_name, save_path=None):
    """绘制网格搜索中给定最优功率下的容量-成本曲线"""
    set_chinese_style()
    _fix_chinese_font()

    if not grid_points:
        return

    df = pd.DataFrame(grid_points)
    # 找最优功率
    best_idx = df['annual_cost'].idxmin()
    best_P = df.loc[best_idx, 'P_ess']

    # 固定该功率，看容量变化
    subset = df[np.abs(df['P_ess'] - best_P) < 1].sort_values('E_ess')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(subset['E_ess'].values, subset['annual_cost'].values / 10000,
            'b-', linewidth=2, marker='o', markersize=4)
    ax.axvline(subset.loc[subset['annual_cost'].idxmin(), 'E_ess'],
               color='r', linestyle='--', alpha=0.7,
               label=f"最优容量: {subset.loc[subset['annual_cost'].idxmin(), 'E_ess']:.0f} kWh")
    ax.set_xlabel('储能容量 (kWh)')
    ax.set_ylabel('年综合成本 (万元)')
    ax.set_title(f'园区 {park_name} (P={best_P:.0f} kW) 容量-成本曲线')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


if __name__ == '__main__':
    print('visualization demo 运行完毕')
