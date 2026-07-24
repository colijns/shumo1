# -*- coding: utf-8 -*-
"""
模块：nonlinear_programming
功能：非线性规划（NLP）求解，基于 scipy.optimize.minimize 的 SLSQP 算法（支持等式/不等式约束与边界）
适用题型：通用高频（带约束的最优设计、经济调度、曲线拟合型优化等）
依赖：numpy, scipy, matplotlib
用法：直接运行 `python nonlinear_programming.py` 查看 demo；或 import 后调用 solve_nlp
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存/跳过绘图
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def solve_nlp(func, x0, bounds=None, constraints=None):
    """
    求解非线性规划：min func(x)，支持边界与非线性约束（SLSQP）。

    参数：
        func        : 目标函数，接收 numpy 向量 x，返回标量
        x0          : 初始点，shape (n,)
        bounds      : 变量边界，如 [(0, 1), (None, None)]，可为 None
        constraints : scipy 风格约束字典或字典列表，例如
                      {'type': 'ineq', 'fun': lambda x: 1 - x[0]**2 - x[1]**2}
                      （ineq 表示 fun(x) >= 0，eq 表示 fun(x) = 0），可为 None

    返回：
        dict，键为：
            'x'       : 最优解向量（失败时为 None）
            'fun'     : 最优目标值（失败时为 None）
            'success' : 是否求解成功（bool）
            'message' : 求解器返回的状态信息
            'nit'     : 迭代次数
    """
    x0 = np.asarray(x0, dtype=float)
    res = minimize(func, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': 500, 'ftol': 1e-10})
    return {
        'x': res.x if res.success else None,
        'fun': float(res.fun) if res.success else None,
        'success': bool(res.success),
        'message': str(res.message),
        'nit': int(res.nit),
    }


def _rosenbrock(x):
    """Rosenbrock 函数（香蕉函数），全局最小值在 (1, 1)，f=0。"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


if __name__ == "__main__":
    print("=" * 60)
    print("非线性规划 demo：圆约束下的 Rosenbrock 函数最小化")
    print("=" * 60)
    # 问题：min (1-x)^2 + 100*(y-x^2)^2
    #       s.t. x^2 + y^2 <= 1   （单位圆内）
    #            x + y >= 0
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] ** 2 - x[1] ** 2},  # 圆内：1-x^2-y^2 >= 0
        {'type': 'ineq', 'fun': lambda x: x[0] + x[1]},                # x+y >= 0
    ]
    bounds = [(-2, 2), (-2, 2)]
    x0 = [0.0, 0.5]                     # 初始点

    res = solve_nlp(_rosenbrock, x0, bounds=bounds, constraints=constraints)

    if res['success']:
        x = res['x']
        print("求解状态：成功")
        print(f"迭代次数：{res['nit']}")
        print(f"最优解：x = {x[0]:.4f}, y = {x[1]:.4f}")
        print(f"最优目标值：f = {res['fun']:.4f}")
        print(f"约束校验：x^2+y^2 = {x[0]**2 + x[1]**2:.4f} <= 1，x+y = {x[0] + x[1]:.4f} >= 0")
    else:
        print(f"求解失败：{res['message']}")

    # 可视化：等高线 + 约束区域 + 最优解
    fig, ax = plt.subplots(figsize=(7, 6))
    xx = np.linspace(-1.5, 1.5, 300)
    yy = np.linspace(-1.5, 1.5, 300)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (1 - XX) ** 2 + 100 * (YY - XX ** 2) ** 2
    # 对数刻度等高线，便于观察香蕉谷
    cs = ax.contour(XX, YY, ZZ, levels=np.logspace(-1, 3, 15), cmap='viridis')
    ax.clabel(cs, inline=True, fontsize=8)
    # 画出单位圆约束边界
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', label='圆约束 $x^2+y^2=1$')
    ax.axline((0, 0), slope=-1, color='orange', linestyle='--', label='约束 $x+y=0$')
    if res['success']:
        ax.plot(x[0], x[1], 'r*', markersize=15, label='约束最优解')
    ax.plot(1, 1, 'b^', markersize=10, label='无约束全局最优 (1,1)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('带约束的 Rosenbrock 函数最小化（SLSQP）')
    ax.legend()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('nlp_demo.png', dpi=150)
    print("已保存约束可视化图：nlp_demo.png")
    plt.close(fig)
