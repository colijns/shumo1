# -*- coding: utf-8 -*-
"""
模块：ode_solver
功能：常微分方程（组）数值求解——scipy.integrate.solve_ivp，含 Logistic 人口、SIR 传染病、Lotka-Volterra 捕食三大经典案例
适用题型：国赛A题高频（连续系统建模、传染病动力学、种群生态）
依赖：numpy, scipy, matplotlib
用法：直接运行 `python ode_solver.py` 查看 demo；或 import 后调用核心函数
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def solve_logistic(r=0.3, K=1000.0, x0=50.0, t_span=(0, 50), n_points=500):
    """Logistic 人口（阻滞增长）模型：dx/dt = r*x*(1 - x/K)。

    参数
    ----
    r : float
        内禀增长率。
    K : float
        环境容纳量（承载力）。
    x0 : float
        初始种群数量。
    t_span : tuple
        求解时间区间 (t_start, t_end)。
    n_points : int
        输出采样点数。

    返回
    ----
    result : dict
        {'t': 时间数组, 'x': 种群数量数组, 'r': r, 'K': K}
    """
    # 右端函数：Logistic 增长 = 指数增长 × 阻滞因子
    def rhs(t, x):
        return [r * x[0] * (1 - x[0] / K)]

    # RK45 自适应步长求解，t_eval 控制输出采样点
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [x0], t_eval=t_eval, dense_output=True)
    return {'t': sol.t, 'x': sol.y[0], 'r': r, 'K': K}


def solve_sir(beta, gamma, S0=0.99, I0=0.01, R0=0.0, t_span=(0, 100), n_points=1000):
    """SIR 传染病模型：dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I。

    参数
    ----
    beta : float
        传染率（有效接触率）。
    gamma : float
        恢复率（1/gamma 为平均传染期）。
    S0, I0, R0 : float
        易感者、感染者、康复者初始比例（总和应为 1）。
    t_span : tuple
        求解时间区间 (t_start, t_end)。
    n_points : int
        输出采样点数。

    返回
    ----
    result : dict
        {'t': 时间数组, 'S', 'I', 'R': 三类人群比例数组,
         'R0_basic': 基本再生数 beta/gamma}
    """
    def rhs(t, y):
        S, I, R = y
        return [-beta * S * I,            # 易感者减少
                beta * S * I - gamma * I,  # 感染者：新增 - 康复
                gamma * I]                 # 康复者增加

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [S0, I0, R0], t_eval=t_eval, dense_output=True)
    S, I, R = sol.y
    return {'t': sol.t, 'S': S, 'I': I, 'R': R, 'R0_basic': beta / gamma}


def solve_lotka_volterra(alpha=1.0, beta=0.1, delta=0.075, gamma=1.5,
                         x0=40.0, y0=9.0, t_span=(0, 30), n_points=1500):
    """Lotka-Volterra 捕食模型：dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y。

    参数
    ----
    alpha : float
        猎物（食饵）自然增长率。
    beta : float
        捕食率（猎物因被捕食而减少的系数）。
    delta : float
        捕食者因捕食猎物而增长的系数。
    gamma : float
        捕食者自然死亡率。
    x0, y0 : float
        猎物与捕食者初始数量。
    t_span : tuple
        求解时间区间 (t_start, t_end)。
    n_points : int
        输出采样点数。

    返回
    ----
    result : dict
        {'t': 时间数组, 'prey': 猎物数量数组, 'predator': 捕食者数量数组,
         'equilibrium': (gamma/delta, alpha/beta) 共存平衡点}
    """
    def rhs(t, z):
        x, y = z
        return [alpha * x - beta * x * y,   # 猎物：增长 - 被捕食
                delta * x * y - gamma * y]  # 捕食者：因捕食增长 - 自然死亡

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [x0, y0], t_eval=t_eval, dense_output=True)
    return {'t': sol.t, 'prey': sol.y[0], 'predator': sol.y[1],
            'equilibrium': (gamma / delta, alpha / beta)}


def plot_logistic(res):
    """画 Logistic 曲线（时序图）并标注环境容纳量 K。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['t'], res['x'], 'b-', lw=2, label=f'种群数量 x(t)（r={res["r"]}）')
    # 水平虚线标注承载力 K，体现「阻滞增长趋于 K」
    ax.axhline(res['K'], color='r', ls='--', lw=1.5, label=f'环境容纳量 K={res["K"]:.0f}')
    ax.set_xlabel('时间 t')
    ax.set_ylabel('种群数量')
    ax.set_title('Logistic 人口（阻滞增长）模型')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sir(res):
    """画 SIR 三类人群比例时序图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['t'], res['S'], 'b-', lw=2, label='易感者 S(t)')
    ax.plot(res['t'], res['I'], 'r-', lw=2, label='感染者 I(t)')
    ax.plot(res['t'], res['R'], 'g-', lw=2, label='康复者 R(t)')
    ax.set_xlabel('时间 t（天）')
    ax.set_ylabel('人群比例')
    ax.set_title(f'SIR 传染病模型（基本再生数 R0 = {res["R0_basic"]:.2f}）')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_lotka_volterra(res):
    """画 Lotka-Volterra 时序图 + 相平面图（捕食者-猎物极限环）。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：时序图，可见猎物峰值领先于捕食者峰值
    ax = axes[0]
    ax.plot(res['t'], res['prey'], 'b-', lw=2, label='猎物 x(t)')
    ax.plot(res['t'], res['predator'], 'r-', lw=2, label='捕食者 y(t)')
    ax.set_xlabel('时间 t')
    ax.set_ylabel('种群数量')
    ax.set_title('Lotka-Volterra 捕食模型：时序图')
    ax.legend()
    ax.grid(alpha=0.3)

    # 右：相平面图，周期解呈闭合极限环；红点为共存平衡点
    ax = axes[1]
    ax.plot(res['prey'], res['predator'], 'k-', lw=1.5)
    eq = res['equilibrium']
    ax.plot(eq[0], eq[1], 'ro', ms=8, label=f'平衡点 ({eq[0]:.1f}, {eq[1]:.1f})')
    ax.plot(res['prey'][0], res['predator'][0], 'go', ms=8, label='初始点')
    ax.set_xlabel('猎物数量 x')
    ax.set_ylabel('捕食者数量 y')
    ax.set_title('相平面图（闭合轨线 = 周期振荡）')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print('=' * 55)
    print('常微分方程 demo：三大经典模型（solve_ivp 求解）')
    print('=' * 55)

    # ---------- 案例 1：Logistic 人口模型 ----------
    print('\n【案例 1】Logistic 人口模型 dx/dt = r*x*(1 - x/K)')
    res1 = solve_logistic(r=0.3, K=1000.0, x0=50.0)
    print(f'  参数：r = 0.3，K = 1000，x0 = 50')
    print(f'  t = {res1["t"][-1]:.0f} 时种群数量 x = {res1["x"][-1]:.4f}（应趋近 K = 1000）')
    print(f'  t = 10 时种群数量 x = {res1["x"][100]:.4f}')
    # ---------- 案例 2：SIR 传染病模型 ----------
    print('\n【案例 2】SIR 传染病模型')
    res2 = solve_sir(beta=0.3, gamma=0.1)
    i_peak = res2['I'].max()
    t_peak = res2['t'][np.argmax(res2['I'])]
    print(f'  参数：beta = 0.3（传染率），gamma = 0.1（恢复率）')
    print(f'  基本再生数 R0 = beta/gamma = {res2["R0_basic"]:.4f}（>1 表示疫情会扩散）')
    print(f'  感染峰值比例 = {i_peak:.4f}，出现在第 {t_peak:.4f} 天')
    print(f'  最终康复（累计感染）比例 R(∞) ≈ {res2["R"][-1]:.4f}')
    # ---------- 案例 3：Lotka-Volterra 捕食模型 ----------
    print('\n【案例 3】Lotka-Volterra 捕食模型')
    res3 = solve_lotka_volterra()
    eq = res3['equilibrium']
    print(f'  参数：alpha=1.0, beta=0.1, delta=0.075, gamma=1.5')
    print(f'  共存平衡点 = ({eq[0]:.4f}, {eq[1]:.4f})')
    print(f'  猎物数量范围：[{res3["prey"].min():.4f}, {res3["prey"].max():.4f}]')
    print(f'  捕食者数量范围：[{res3["predator"].min():.4f}, {res3["predator"].max():.4f}]')
    print('  相平面图为闭合轨线，说明两种群呈周期振荡')

    fig1 = plot_logistic(res1)
    fig1.savefig('07_simulation_ode/ode_logistic.png', dpi=150)
    plt.close(fig1)
    fig2 = plot_sir(res2)
    fig2.savefig('07_simulation_ode/ode_sir.png', dpi=150)
    plt.close(fig2)
    fig3 = plot_lotka_volterra(res3)
    fig3.savefig('07_simulation_ode/ode_lotka_volterra.png', dpi=150)
    plt.close(fig3)
    print('\n三个案例的图均已保存。')
