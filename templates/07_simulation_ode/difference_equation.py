# -*- coding: utf-8 -*-
"""
模块：difference_equation
功能：差分方程应用——房贷等额本息还款计算、离散 Logistic 种群模型（含随 r 变化的混沌演示与分岔图）
适用题型：通用（经济金融建模、离散动力系统、混沌现象分析）
依赖：numpy, pandas, matplotlib
用法：直接运行 `python difference_equation.py` 查看 demo；或 import 后调用核心函数
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def equal_installment(principal, annual_rate, n_months):
    """等额本息还款计算（差分方程 B_{k+1} = B_k*(1+i) - A 的显式求解）。

    参数
    ----
    principal : float
        贷款本金（元）。
    annual_rate : float
        年利率（小数形式，如 0.049 表示 4.9%）。
    n_months : int
        还款期数（月）。

    返回
    ----
    result : dict
        {'monthly_payment': float 每月还款额,
         'total_payment': float 还款总额,
         'total_interest': float 利息总额,
         'schedule': pandas.DataFrame 还款计划表
                     （列：月份、月供、当月利息、当月本金、剩余本金）}
    """
    i = annual_rate / 12  # 月利率
    n = n_months
    # 等额本息月供公式：A = P * i * (1+i)^n / ((1+i)^n - 1)
    factor = (1 + i) ** n
    A = principal * i * factor / (factor - 1)

    # 用差分方程逐月递推余额，同时拆分每月利息与本金
    rows = []
    balance = principal
    for month in range(1, n + 1):
        interest = balance * i          # 当月利息 = 剩余本金 × 月利率
        principal_part = A - interest   # 当月偿还本金 = 月供 - 当月利息
        balance = balance - principal_part  # 差分递推：B_{k+1} = B_k*(1+i) - A
        rows.append([month, A, interest, principal_part, max(balance, 0.0)])

    schedule = pd.DataFrame(rows, columns=['月份', '月供', '当月利息', '当月本金', '剩余本金'])
    return {'monthly_payment': A,
            'total_payment': A * n,
            'total_interest': A * n - principal,
            'schedule': schedule}


def discrete_logistic(r, x0, n):
    """离散 Logistic 映射迭代：x_{k+1} = r * x_k * (1 - x_k)。

    参数
    ----
    r : float
        增长参数（0 < r <= 4；r 增大依次出现稳定点、周期倍化、混沌）。
    x0 : float
        初始值（0 < x0 < 1）。
    n : int
        迭代步数。

    返回
    ----
    x : numpy.ndarray
        长度为 n+1 的迭代序列（含 x0）。
    """
    x = np.empty(n + 1)
    x[0] = x0
    # 逐步迭代差分方程
    for k in range(n):
        x[k + 1] = r * x[k] * (1 - x[k])
    return x


def plot_logistic_sequences(r_list, x0=0.2, n=100):
    """画不同 r 取值下离散 Logistic 迭代序列对比图（展示混沌演化）。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in r_list:
        x = discrete_logistic(r, x0, n)
        ax.plot(range(n + 1), x, '.-', ms=4, lw=1, label=f'r = {r}')
    ax.set_xlabel('迭代步数 k')
    ax.set_ylabel('x_k')
    ax.set_title('离散 Logistic 模型：不同 r 的迭代行为（收敛 → 周期 → 混沌）')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bifurcation(r_min=2.5, r_max=4.0, n_r=800, n_transient=300, n_keep=100, x0=0.2):
    """画离散 Logistic 分岔图：横轴 r，纵轴为稳态后的 x 取值分布。

    参数
    ----
    r_min, r_max : float
        参数 r 扫描范围。
    n_r : int
        r 采样点数。
    n_transient : int
        丢弃的瞬态迭代步数（让轨道进入吸引子）。
    n_keep : int
        保留绘制的稳态迭代步数。
    x0 : float
        初始值。

    返回
    ----
    fig : matplotlib.figure.Figure
        画好的图对象。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    r_values = np.linspace(r_min, r_max, n_r)
    for r in r_values:
        # 先迭代 n_transient 步丢弃瞬态，再保留后 n_keep 步作为稳态采样
        x = discrete_logistic(r, x0, n_transient + n_keep)[-n_keep:]
        ax.plot(np.full_like(x, r), x, ',k', alpha=0.5)  # ',' 为极细像素点
    ax.set_xlabel('增长参数 r')
    ax.set_ylabel('稳态 x 取值')
    ax.set_title('离散 Logistic 分岔图（周期倍化通向混沌）')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print('=' * 55)
    print('差分方程 demo：房贷等额本息 + 离散 Logistic 混沌')
    print('=' * 55)

    # ---------- 案例 1：房贷等额本息 ----------
    print('\n【案例 1】房贷等额本息还款计算')
    principal, annual_rate, n_months = 1_000_000.0, 0.049, 240  # 100 万、4.9%、20 年
    res = equal_installment(principal, annual_rate, n_months)
    print(f'  贷款本金：{principal:,.2f} 元，年利率 {annual_rate * 100:.2f}%，期限 {n_months} 个月')
    print(f'  每月还款额：{res["monthly_payment"]:.4f} 元')
    print(f'  还款总额：  {res["total_payment"]:.4f} 元')
    print(f'  利息总额：  {res["total_interest"]:.4f} 元')
    print('  还款计划表前 5 行：')
    print(res['schedule'].head().to_string(index=False,
          float_format=lambda v: f'{v:.4f}'))
    print(f'  期末剩余本金：{res["schedule"]["剩余本金"].iloc[-1]:.4f} 元（应≈0，贷款结清）')

    # 画剩余本金曲线 + 月供中利息/本金占比堆叠图（抽样每 12 个月显示）
    sch = res['schedule']
    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sch['月份'], sch['剩余本金'], 'b-', lw=2)
    axes[0].set_xlabel('月份')
    axes[0].set_ylabel('剩余本金（元）')
    axes[0].set_title('等额本息：剩余本金变化曲线')
    axes[0].grid(alpha=0.3)
    idx = np.arange(0, len(sch), 12)  # 每年抽 1 个月显示，避免过密
    axes[1].bar(sch['月份'][idx], sch['当月利息'][idx], label='当月利息', color='salmon')
    axes[1].bar(sch['月份'][idx], sch['当月本金'][idx], bottom=sch['当月利息'][idx],
                label='当月本金', color='steelblue')
    axes[1].set_xlabel('月份')
    axes[1].set_ylabel('月供构成（元）')
    axes[1].set_title('月供构成：前期利息占比高，后期本金占比高')
    axes[1].legend()
    fig1.tight_layout()

    # ---------- 案例 2：离散 Logistic 模型（含混沌演示） ----------
    print('\n【案例 2】离散 Logistic 模型 x_{k+1} = r*x_k*(1-x_k)')
    # r=2.5 收敛到不动点；r=3.2 周期 2 振荡；r=3.9 进入混沌
    for r in [2.5, 3.2, 3.9]:
        x = discrete_logistic(r, x0=0.2, n=100)
        tail = np.round(x[-4:], 4)
        print(f'  r = {r}: 末 4 步取值 {tail}'
              + ('（收敛于不动点）' if r < 3 else '（周期/混沌振荡）'))
    fig2 = plot_logistic_sequences([2.5, 3.2, 3.9])
    fig3 = plot_bifurcation()
    print('  分岔图已绘制：可观察到 1→2→4→… 周期倍化最终进入混沌')

    fig1.savefig('07_simulation_ode/difference_loan.png', dpi=150)
    plt.close(fig1)
    fig2.savefig('07_simulation_ode/difference_logistic_sequences.png', dpi=150)
    plt.close(fig2)
    fig3.savefig('07_simulation_ode/difference_bifurcation.png', dpi=150)
    plt.close(fig3)
    print('\n全部图已保存。')
