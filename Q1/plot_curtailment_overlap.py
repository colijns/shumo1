# -*- coding: utf-8 -*-
"""
图2 各园区50kW/100kWh储能工况下逐时弃电功率与储能充放电功率叠合对比图

拆分为三张独立小图（每个园区一张），验证"弃电规模与成本改善幅度正相关"：
  - 红色粗实线：逐时弃电功率 q(t) = P_curt_pv + P_curt_w（有储能后残余弃电，
                仅风光出力大于负荷时数值大于0，其余时刻归零）
  - 蓝色细虚线：储能充电功率 pc(t) = P_ch（仅弃电时段存在，峰值≤50kW）
  - 橙色点划线：储能放电功率 pd(t) = P_dis（仅负荷缺口时段存在，峰值≤50kW）
  - 绿色实线（右轴）：储能荷电状态 SOC(t) = E[t]/E_ess（百分比 0~100%，
                长度25点 t=0..24，日末 E[24]=E[0] 回归初始荷电）

硬性约束：同一时刻蓝、橙曲线不同时出现（MILP 充放电互斥）；充电曲线不在无弃电
时段产生数值。数据口径仅采用第二小问固定 50kW/100kWh 储能优化调度结果，
不混入无储能/最优配置数据。

用法（math 环境，UTF-8）：
  $env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'
  & "E:\\Software\\Scoop\\apps\\miniconda3\\current\\envs\\math\\python.exe" Q1/plot_curtailment_overlap.py
"""

import sys
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'templates'))
from common.plot_style import set_chinese_style, save_fig

sys.path.append(os.path.dirname(__file__))
from data_loader import load_load_data, load_solar_wind_data
from milp_model import run_no_storage, run_fixed_storage


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
PARKS = ['A', 'B', 'C']
P_ESS, E_ESS = 50, 100  # 第二小问固定方案

# ---- 统一坐标轴规范（三子图完全一致）----
X_TICKS = np.arange(0, 25, 2)          # 0,2,4,...,24
Y_LIM = (0, 280)                       # 容纳峰值247kW(园区C)
Y_TICKS = np.arange(0, 281, 40)        # 0,40,80,...,280

# ---- 曲线样式（全局统一图例）----
STYLE_CURT = dict(color='#d62728', linewidth=2.2, linestyle='-')    # 红色粗实线
STYLE_CH   = dict(color='#1f77b4', linewidth=1.5, linestyle='--')   # 蓝色细虚线
STYLE_DIS  = dict(color='#ff7f0e', linewidth=1.5, linestyle='-.')   # 橙色点划线
STYLE_SOC  = dict(color='#2ca02c', linewidth=1.8, linestyle='-')    # 绿色实线（SOC，右轴）


def _fix_chinese_font():
    """在 set_chinese_style 之后确保中文字体生效（与 visualization.py 一致）"""
    for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            _name = font_manager.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


def _step_series(y):
    """将 24 点逐时序列扩展为 steps-post 阶梯坐标，使曲线覆盖 [0,24] 全区间。

    末尾追加 t=24 处最后一个值，配合 drawstyle='steps-post' 让 t=23 时段的功率
    延伸到 24，符合"时刻 t 范围 0–24h"的坐标规范。
    """
    x = np.arange(24)
    x2 = np.append(x, 24.0)
    y2 = np.append(np.asarray(y, dtype=float), float(y[-1]))
    return x2, y2


def _plot_one_park(ax, park, L, P, W):
    """在给定 ax 上绘制单个园区的弃电与充放电叠合曲线，返回是否求解成功。"""
    print(f'求解 园区{park}：无储能 + {P_ESS}kW/{E_ESS}kWh ...')
    r_no = run_no_storage(L, P, W)
    r_fix = run_fixed_storage(L, P, W, P_ess=P_ESS, E_ess=E_ESS)

    if not (r_no['success'] and r_fix['success']):
        ax.set_title(f'园区{park} 求解失败: '
                     f'{r_no.get("status")}/{r_fix.get("status")}')
        return False

    curt = np.array(r_fix['P_curt_pv']) + np.array(r_fix['P_curt_w'])  # 逐时弃电
    p_ch = np.array(r_fix['P_ch'])                                     # 逐时充电
    p_dis = np.array(r_fix['P_dis'])                                   # 逐时放电

    # ---- 绘制顺序：底层红弃电 -> 中层蓝充电 -> 上层橙放电 ----
    xc, yc = _step_series(curt)
    ax.plot(xc, yc, drawstyle='steps-post', label='逐时弃电功率', **STYLE_CURT)
    xh, yh = _step_series(p_ch)
    ax.plot(xh, yh, drawstyle='steps-post', label='储能充电功率', **STYLE_CH)
    xd, yd = _step_series(p_dis)
    ax.plot(xd, yd, drawstyle='steps-post', label='储能放电功率', **STYLE_DIS)

    ax.axhline(0, color='k', linewidth=0.8)

    # ---- 右上角定量文本框（三行，横向对比论证）----
    q_total = float(curt.sum())                                  # 全日总弃电量 kWh
    ch_total = float(p_ch.sum())                                 # 储能日充电总量 kWh
    cost_delta = r_no['daily_cost'] - r_fix['daily_cost']        # 日运行成本降幅 元
    txt = (f'全日总弃电量：{q_total:.1f} kWh\n'
           f'储能日充电总量：{ch_total:.1f} kWh\n'
           f'日运行成本降幅：{cost_delta:.2f} 元')
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.45', fc='white',
                      ec='#888888', alpha=0.9))

    ax.set_ylabel('功率 (kW)', fontsize=10)
    ax.set_ylim(Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.set_xlim(0, 24)
    ax.set_xticks(X_TICKS)
    ax.set_xlabel('时刻 t (h)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- 储能 SOC 曲线（右轴百分比，状态轨迹）----
    # SOC(t) = E[t] / E_ess * 100；E 长度 25（t=0..24），折线连接各整数时刻
    # 荷电状态；日末 E[24]=E[0] 回归初始荷电。
    e_ess_val = r_fix['E_ess']
    ax2 = None
    if e_ess_val > 1e-6:
        soc = np.array(r_fix['E']) / e_ess_val * 100.0
        xs = np.arange(len(soc))
        ax2 = ax.twinx()
        ax2.plot(xs, soc, label='储能SOC', **STYLE_SOC)
        ax2.set_ylabel('储能 SOC (%)', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.set_yticks(np.arange(0, 101, 20))
        ax2.tick_params(axis='y', labelsize=9)

    # 图例置于图下方，合并功率轴与 SOC 轴句柄
    h1, l1 = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        h1, l1 = h1 + h2, l1 + l2
    ax.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, -0.17),
              ncol=4, frameon=False, fontsize=10)
    return True


def plot_curtailment_overlap(save_dir=None):
    """绘制三张独立小图（园区 A/B/C 各一张），返回已保存路径列表。"""
    set_chinese_style()
    _fix_chinese_font()

    print('读取数据...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    saved = []
    for park in PARKS:
        L = load_data[park]
        P = G_pv_data[park]
        W = G_w_data[park]

        fig, ax = plt.subplots(figsize=(9, 5.2))
        _plot_one_park(ax, park, L, P, W)

        fig.suptitle(
            f'图2 园区{park}（{P_ESS}kW/{E_ESS}kWh）储能工况下'
            f'逐时弃电功率与储能充放电功率叠合对比图',
            fontsize=12, y=0.97)
        fig.subplots_adjust(top=0.88, bottom=0.22, left=0.085, right=0.91)

        if save_dir:
            out = os.path.join(save_dir, f'结论2_弃电与充放电叠合图_园区{park}.png')
            save_fig(fig, out)
            saved.append(out)
        plt.close(fig)

    return saved


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = plot_curtailment_overlap(save_dir=OUTPUT_DIR)
    for p in paths:
        print(f'已保存：{p}')
    print('plot_curtailment_overlap demo 运行完毕')
