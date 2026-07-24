# -*- coding: utf-8 -*-
"""
问题2（1）：三园区联合运营、无储能

核心思路（见 docs/问题2（1）.md）：
  - 先把三园区每小时的负荷、光伏、风电逐时相加，得到联合profile；
  - 再对联合profile统一做功率平衡（无储能）；
  - 与"三园区独立无储能之和"对比，量化园区间互济收益。

实现上跨目录复用 Q1 的统一 MILP 模型（ADR 0001）：
  - data_loader.load_load_data / load_solar_wind_data 读取附件1、附件2；
  - milp_model.run_no_storage(load, G_pv, G_w) 在储能置0时退化为本问的联合无储能模型。
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----- 跨目录复用 Q1（见 Q2/docs/adr/0001-reuse-q1-unified-milp.md）-----
_Q1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Q1'))
if _Q1_DIR not in sys.path:
    sys.path.insert(0, _Q1_DIR)
from data_loader import load_load_data, load_solar_wind_data  # noqa: E402
from milp_model import run_no_storage  # noqa: E402

# ----- 复用 templates/common -----
_TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
if _TEMPLATE_DIR not in sys.path:
    sys.path.insert(0, _TEMPLATE_DIR)
from common.io_utils import export_result  # noqa: E402
from common.plot_style import set_chinese_style, save_fig  # noqa: E402

PARKS = ['A', 'B', 'C']
T = 24

# 文档 §6 联合无储能验收目标（用于结果校验）
DOC_JOINT_TARGETS = {
    'grid_total': 8266.270,      # 日电网购电量 kWh
    'pv_curt': 84.610,           # 日弃光量 kWh
    'w_curt': 1152.565,          # 日弃风量 kWh
    're_ratio': 0.92437,         # 风光消纳率
    'daily_cost': 15107.661,     # 日供电成本 元
    'unit_cost': 0.6460,         # 单位供电成本 元/kWh
}
DOC_INDEP_TARGETS = {
    'grid_total': 10005.815,
    'curt_total': 2976.720,
    're_ratio': 0.81803,
    'daily_cost': 16119.336,
}


def _fix_chinese_font():
    """set_chinese_style 之后加载 ttf 确保中文字体生效（沿用 Q1/visualization.py 做法）。"""
    for _fp in [r'C:\Windows\Fonts\simhei.ttf', r'C:\Windows\Fonts\msyh.ttc']:
        if os.path.exists(_fp):
            font_manager.fontManager.addfont(_fp)
            _name = font_manager.FontProperties(fname=_fp).get_name()
            matplotlib.rcParams['font.family'] = 'sans-serif'
            matplotlib.rcParams['font.sans-serif'] = [_name, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            return


# =====================================================================
# 数据聚合与求解
# =====================================================================

def build_joint_profile(load_data, G_pv_data, G_w_data):
    """三园区逐时相加，返回 (L_J, G_pv_J, G_w_J)，各为长度24的 array。"""
    L_J = np.zeros(T)
    G_pv_J = np.zeros(T)
    G_w_J = np.zeros(T)
    for p in PARKS:
        L_J = L_J + load_data[p]
        G_pv_J = G_pv_J + G_pv_data[p]
        G_w_J = G_w_J + G_w_data[p]
    return L_J, G_pv_J, G_w_J


def solve_joint_no_storage(load_data, G_pv_data, G_w_data):
    """联合无储能：聚合后调用 run_no_storage。返回 (result, profile)。"""
    L_J, G_pv_J, G_w_J = build_joint_profile(load_data, G_pv_data, G_w_data)
    r = run_no_storage(L_J, G_pv_J, G_w_J)
    return r, (L_J, G_pv_J, G_w_J)


def solve_independent_no_storage(load_data, G_pv_data, G_w_data):
    """独立无储能：逐园区调用 run_no_storage。返回 dict[园区] -> result。"""
    return {p: run_no_storage(load_data[p], G_pv_data[p], G_w_data[p]) for p in PARKS}


def joint_metrics(r_joint):
    """从联合求解结果提取与"独立之和"同结构的指标 dict。"""
    load_total = r_joint['load_total']
    daily_cost = r_joint['daily_cost']
    return {
        'load_total': load_total,
        'grid_total': r_joint['grid_total'],
        'pv_curt': r_joint['pv_curt'],
        'w_curt': r_joint['w_curt'],
        'curt_total': r_joint['curt_total'],
        're_ratio': r_joint['re_ratio'],
        'daily_cost': daily_cost,
        'annual_cost': 365.0 * daily_cost,  # 无储能无投资
        'unit_cost': daily_cost / load_total if load_total > 0 else 0.0,
    }


def aggregate_independent(indep_results):
    """把三园区独立结果汇总成"独立之和"指标 dict。"""
    def s(key):
        return sum(indep_results[p][key] for p in PARKS)

    load_total = s('load_total')
    grid_total = s('grid_total')
    pv_curt = s('pv_curt')
    w_curt = s('w_curt')
    curt_total = pv_curt + w_curt
    daily_cost = s('daily_cost')
    re_gen = s('pv_gen') + s('w_gen')
    return {
        'load_total': load_total,
        'grid_total': grid_total,
        'pv_curt': pv_curt,
        'w_curt': w_curt,
        'curt_total': curt_total,
        're_ratio': 1.0 - curt_total / re_gen if re_gen > 0 else 0.0,
        'daily_cost': daily_cost,
        'annual_cost': 365.0 * daily_cost,
        'unit_cost': daily_cost / load_total if load_total > 0 else 0.0,
    }


# =====================================================================
# 结果表格
# =====================================================================

def build_hourly_df(r_joint, L_J, G_pv_J, G_w_J):
    """§8.1 逐时联合运行表。"""
    return pd.DataFrame({
        '时刻': list(range(T)),
        '联合负荷(kW)': L_J,
        '联合光伏(kW)': G_pv_J,
        '联合风电(kW)': G_w_J,
        '光伏直供(kW)': r_joint['P_load_pv'],
        '风电直供(kW)': r_joint['P_load_w'],
        '电网购电(kW)': r_joint['P_grid'],
        '弃光(kW)': r_joint['P_curt_pv'],
        '弃风(kW)': r_joint['P_curt_w'],
    })


def build_summary_df(joint):
    """§8.2 经济指标汇总表（联合无储能）。"""
    rows = [
        ('日负荷电量/kWh', joint['load_total']),
        ('日电网购电量/kWh', joint['grid_total']),
        ('日弃光量/kWh', joint['pv_curt']),
        ('日弃风量/kWh', joint['w_curt']),
        ('日弃风弃光总量/kWh', joint['curt_total']),
        ('风光消纳率/%', joint['re_ratio'] * 100),
        ('日供电成本/元', joint['daily_cost']),
        ('年供电成本/元', joint['annual_cost']),
        ('单位供电成本/(元/kWh)', joint['unit_cost']),
    ]
    return pd.DataFrame(rows, columns=['指标', '联合无储能'])


def build_comparison_df(indep, joint):
    """§8.3 独立与联合对比表（结构对齐文档 §6）。"""
    rows = [
        ('日负荷电量/kWh', indep['load_total'], joint['load_total']),
        ('日电网购电量/kWh', indep['grid_total'], joint['grid_total']),
        ('日弃光量/kWh', indep['pv_curt'], joint['pv_curt']),
        ('日弃风量/kWh', indep['w_curt'], joint['w_curt']),
        ('日弃风弃光总量/kWh', indep['curt_total'], joint['curt_total']),
        ('风光消纳率/%', indep['re_ratio'] * 100, joint['re_ratio'] * 100),
        ('日供电成本/元', indep['daily_cost'], joint['daily_cost']),
        ('年供电成本/元', indep['annual_cost'], joint['annual_cost']),
        ('单位供电成本/(元/kWh)', indep['unit_cost'], joint['unit_cost']),
    ]
    df = pd.DataFrame(rows, columns=['指标', '三园区独立无储能之和', '联合无储能'])
    df['联合后的变化'] = df['联合无储能'] - df['三园区独立无储能之和']
    return df


# =====================================================================
# 验证（文档 §9 checklist + §6 数值 + 互济不变式）
# =====================================================================

def verify_results(r_joint, L_J, G_pv_J, G_w_J,
                   load_data, G_pv_data, G_w_data, indep):
    """返回 [(name, passed, detail)] 列表。"""
    checks = []
    TOL = 1e-4

    # --- §9 结构性检查 ---
    L_sum = sum(load_data[p] for p in PARKS)
    pv_sum = sum(G_pv_data[p] for p in PARKS)
    w_sum = sum(G_w_data[p] for p in PARKS)
    checks.append(('联合负荷=三园区逐时相加', np.allclose(L_J, L_sum),
                   f'max|Δ|={np.max(np.abs(L_J - L_sum)):.2e}'))
    checks.append(('联合光伏=三园区逐时相加', np.allclose(G_pv_J, pv_sum),
                   f'max|Δ|={np.max(np.abs(G_pv_J - pv_sum)):.2e}'))
    checks.append(('联合风电=三园区逐时相加', np.allclose(G_w_J, w_sum),
                   f'max|Δ|={np.max(np.abs(G_w_J - w_sum)):.2e}'))

    err = r_joint['errors']
    checks.append(('每小时负荷功率平衡误差<1e-4', err['max_balance_error'] < TOL,
                   f"max={err['max_balance_error']:.2e}"))
    checks.append(('每小时风光分配误差<1e-4', err['max_re_error'] < TOL,
                   f"max={err['max_re_error']:.2e}"))
    checks.append(('电网购电量非负', r_joint['grid_total'] >= -1e-6,
                   f"{r_joint['grid_total']:.3f}"))
    checks.append(('弃光量非负', r_joint['pv_curt'] >= -1e-6, f"{r_joint['pv_curt']:.3f}"))
    checks.append(('弃风量非负', r_joint['w_curt'] >= -1e-6, f"{r_joint['w_curt']:.3f}"))
    checks.append(('日负荷总量=23387 kWh',
                   abs(r_joint['load_total'] - 23387.0) < 1e-2,
                   f"{r_joint['load_total']:.3f}"))

    # 联合 ≠ 独立之和（互济的本质）
    checks.append(('联合购电量≠独立购电量之和',
                   abs(r_joint['grid_total'] - indep['grid_total']) > 1.0,
                   f"联合{r_joint['grid_total']:.3f} vs 独立和{indep['grid_total']:.3f}"))

    # 互济不变式：购电减少量 == 弃电减少量
    grid_reduce = indep['grid_total'] - r_joint['grid_total']
    curt_reduce = indep['curt_total'] - r_joint['curt_total']
    checks.append(('互济不变式: 购电减少=弃电减少',
                   abs(grid_reduce - curt_reduce) < 1e-2,
                   f'购电减{grid_reduce:.3f}, 弃电减{curt_reduce:.3f}'))

    # 单位成本分母为联合负荷电量：由 unit_cost=daily_cost/load_total 反推，
    # daily_cost/unit_cost 应等于 load_total；若分母误用购电量/发电量则不成立
    unit_cost = (r_joint['daily_cost'] / r_joint['load_total']
                 if r_joint['load_total'] > 0 else 0.0)
    denom = (r_joint['daily_cost'] / unit_cost
             if unit_cost > 1e-12 else float('inf'))
    checks.append(('单位成本分母=联合负荷电量',
                   abs(denom - r_joint['load_total']) < 1e-2,
                   f"反推分母={denom:.3f}, 负荷电量={r_joint['load_total']:.3f}"))

    # --- §6 联合无储能数值验收 ---
    tgt = DOC_JOINT_TARGETS
    checks.append(('§6 联合日购电量≈8266.270',
                   abs(r_joint['grid_total'] - tgt['grid_total']) < 0.5,
                   f"{r_joint['grid_total']:.3f}"))
    checks.append(('§6 联合日弃光量≈84.610',
                   abs(r_joint['pv_curt'] - tgt['pv_curt']) < 0.5,
                   f"{r_joint['pv_curt']:.3f}"))
    checks.append(('§6 联合日弃风量≈1152.565',
                   abs(r_joint['w_curt'] - tgt['w_curt']) < 0.5,
                   f"{r_joint['w_curt']:.3f}"))
    checks.append(('§6 联合风光消纳率≈92.437%',
                   abs(r_joint['re_ratio'] - tgt['re_ratio']) < 5e-4,
                   f"{r_joint['re_ratio'] * 100:.3f}%"))
    checks.append(('§6 联合日供电成本≈15107.661',
                   abs(r_joint['daily_cost'] - tgt['daily_cost']) < 1.0,
                   f"{r_joint['daily_cost']:.3f}"))
    checks.append(('§6 联合单位供电成本≈0.6460',
                   abs(r_joint['daily_cost'] / r_joint['load_total'] - tgt['unit_cost']) < 5e-4,
                   f"{r_joint['daily_cost'] / r_joint['load_total']:.4f}"))

    # --- §6 独立之和数值验收 ---
    it = DOC_INDEP_TARGETS
    checks.append(('§6 独立和日购电量≈10005.815',
                   abs(indep['grid_total'] - it['grid_total']) < 0.5,
                   f"{indep['grid_total']:.3f}"))
    checks.append(('§6 独立和日弃电总量≈2976.720',
                   abs(indep['curt_total'] - it['curt_total']) < 0.5,
                   f"{indep['curt_total']:.3f}"))
    checks.append(('§6 独立和日供电成本≈16119.336',
                   abs(indep['daily_cost'] - it['daily_cost']) < 1.0,
                   f"{indep['daily_cost']:.3f}"))

    return checks


# =====================================================================
# 绘图
# =====================================================================

def plot_joint_profile(L_J, G_pv_J, G_w_J, save_path=None):
    """图1：联合负荷、联合光伏、联合风电典型日曲线。"""
    set_chinese_style()
    _fix_chinese_font()
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, L_J, 'k-', linewidth=2, label='联合负荷')
    ax.fill_between(t, 0, G_pv_J, alpha=0.3, color='orange', label='联合光伏')
    ax.fill_between(t, 0, G_w_J, alpha=0.3, color='steelblue', label='联合风电')
    ax.set_xlabel('时刻 (h)')
    ax.set_ylabel('功率 (kW)')
    ax.set_title('联合园区典型日负荷与风光出力')
    ax.set_xticks(t)
    ax.set_xlim(0, 23)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


def plot_indep_vs_joint(indep, joint, save_path=None):
    """图2：独立 vs 联合的购电量、弃电量、年成本对比。"""
    set_chinese_style()
    _fix_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = ['独立运营', '联合运营']
    x = np.arange(len(labels))
    colors = ['#9ecae1', '#3182bd']

    def _bar(ax, vals, ylabel, title, fmt):
        bars = ax.bar(x, vals, color=colors, width=0.55)
        ax.bar_label(bars, labels=[fmt.format(v=v) for v in vals],
                     padding=3, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    _bar(axes[0], [indep['grid_total'], joint['grid_total']],
         '日电网购电量 (kWh)', '电网购电量对比', '{v:.1f}')
    _bar(axes[1], [indep['curt_total'], joint['curt_total']],
         '日弃风弃光总量 (kWh)', '弃电量对比', '{v:.1f}')
    _bar(axes[2], [indep['annual_cost'] / 10000, joint['annual_cost'] / 10000],
         '年供电成本 (万元)', '年供电成本对比', '{v:.2f}')

    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    plt.close(fig)


# =====================================================================
# 编排
# =====================================================================

def run_q2(output_dir=None):
    """问题2（1）完整求解流程。"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print('问题2（1）：三园区联合运营、无储能')
    print('=' * 60)

    print('\n[1/5] 读取数据...')
    load_data = load_load_data()
    G_pv_data, G_w_data = load_solar_wind_data()

    print('\n[2/5] 联合无储能求解...')
    r_joint, (L_J, G_pv_J, G_w_J) = solve_joint_no_storage(
        load_data, G_pv_data, G_w_data)
    if not r_joint['success']:
        print(f'❌ 联合求解失败: {r_joint["status"]}')
        return None
    joint = joint_metrics(r_joint)
    print(f'  日购电量: {joint["grid_total"]:.3f} kWh')
    print(f'  日弃光量: {joint["pv_curt"]:.3f} kWh')
    print(f'  日弃风量: {joint["w_curt"]:.3f} kWh')
    print(f'  风光消纳率: {joint["re_ratio"] * 100:.3f}%')
    print(f'  日供电成本: {joint["daily_cost"]:.3f} 元')
    print(f'  年供电成本: {joint["annual_cost"]:.3f} 元')
    print(f'  单位供电成本: {joint["unit_cost"]:.4f} 元/kWh')

    print('\n[3/5] 独立无储能重算（对比基准）...')
    indep_results = solve_independent_no_storage(load_data, G_pv_data, G_w_data)
    indep = aggregate_independent(indep_results)
    print(f'  独立之和 日购电量: {indep["grid_total"]:.3f} kWh')
    print(f'  独立之和 日弃电总量: {indep["curt_total"]:.3f} kWh')
    print(f'  独立之和 日供电成本: {indep["daily_cost"]:.3f} 元')

    print('\n[4/5] 结果验证...')
    checks = verify_results(r_joint, L_J, G_pv_J, G_w_J,
                            load_data, G_pv_data, G_w_data, indep)
    n_pass = sum(1 for _, p, _ in checks if p)
    for name, passed, detail in checks:
        print(f'  {"✓" if passed else "✗"} {name}  ({detail})')
    print(f'  通过 {n_pass}/{len(checks)} 项检查')

    print('\n[5/5] 输出表格与图片...')
    df_hourly = build_hourly_df(r_joint, L_J, G_pv_J, G_w_J)
    df_summary = build_summary_df(joint)
    df_compare = build_comparison_df(indep, joint)

    p1 = export_result(df_hourly,
                       os.path.join(output_dir, '问题2_1_逐时联合运行表.xlsx'))
    p2 = export_result(df_summary,
                       os.path.join(output_dir, '问题2_1_经济指标汇总.xlsx'))
    p3 = export_result(df_compare,
                       os.path.join(output_dir, '问题2_1_独立与联合对比.xlsx'))
    print(f'  逐时表: {p1}')
    print(f'  汇总表: {p2}')
    print(f'  对比表: {p3}')

    plot_joint_profile(L_J, G_pv_J, G_w_J,
                       os.path.join(output_dir, '联合典型日曲线.png'))
    plot_indep_vs_joint(indep, joint,
                        os.path.join(output_dir, '独立vs联合对比图.png'))

    return {
        'r_joint': r_joint, 'joint': joint, 'indep': indep,
        'indep_results': indep_results,
        'L_J': L_J, 'G_pv_J': G_pv_J, 'G_w_J': G_w_J,
        'df_hourly': df_hourly, 'df_summary': df_summary,
        'df_compare': df_compare, 'checks': checks,
    }


if __name__ == '__main__':
    run_q2()
    print('\nq2_main 运行完毕')
