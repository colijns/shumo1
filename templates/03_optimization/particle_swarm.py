# -*- coding: utf-8 -*-
"""
模块：particle_swarm
功能：手写粒子群优化算法（PSO），求解连续函数全局最小化
适用题型：通用高频（复杂非线性全局优化、参数整定、无梯度问题）
依赖：numpy, matplotlib（不依赖任何第三方优化库）
用法：直接运行 `python particle_swarm.py` 查看 demo；或 import 后调用 pso_minimize
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def pso_minimize(func, dim, bounds, swarm_size=40, max_iter=200, w=0.7, c1=1.5, c2=1.5, seed=None):
    """
    用粒子群算法求 func 的最小值。

    参数：
        func       : 目标函数，接收 shape (dim,) 的向量，返回标量
        dim        : 决策变量维数
        bounds     : 变量边界，shape (dim, 2)，每行为 (下界, 上界)
        swarm_size : 粒子个数
        max_iter   : 最大迭代次数
        w          : 惯性权重
        c1         : 个体学习因子（向自身历史最优靠拢）
        c2         : 社会学习因子（向全局最优靠拢）
        seed       : 随机种子，int 或 None（None 表示不固定种子）

    返回：
        dict，键为：
            'best_x'  : 全局最优位置（numpy 数组，shape (dim,)）
            'best_f'  : 全局最优目标值（float）
            'history' : 每代全局最优目标值列表（长度 max_iter）
    """
    if seed is not None:
        np.random.seed(seed)
    bounds = np.asarray(bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb

    # 1. 初始化粒子位置与速度（速度范围取边界跨度的 10%）
    pos = lb + np.random.rand(swarm_size, dim) * span
    vel = (np.random.rand(swarm_size, dim) - 0.5) * 0.2 * span

    # 2. 个体历史最优与全局最优
    pbest_pos = pos.copy()
    pbest_val = np.array([func(p) for p in pos])
    g_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    history = []
    for it in range(max_iter):
        # 3. 速度更新：惯性 + 个体认知 + 社会认知
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        # 速度限幅，防止飞出搜索空间
        v_max = 0.2 * span
        vel = np.clip(vel, -v_max, v_max)

        # 4. 位置更新并做边界截断
        pos = np.clip(pos + vel, lb, ub)

        # 5. 评价并更新个体/全局最优
        val = np.array([func(p) for p in pos])
        better = val < pbest_val
        pbest_pos[better] = pos[better]
        pbest_val[better] = val[better]
        g_idx = np.argmin(pbest_val)
        if pbest_val[g_idx] < gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()

        history.append(gbest_val)

    return {
        'best_x': gbest_pos,
        'best_f': gbest_val,
        'history': history,
    }


def ackley(x):
    """Ackley 测试函数：多峰，全局最小值在原点，f(0)=0。"""
    x = np.asarray(x)
    n = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


if __name__ == "__main__":
    print("=" * 60)
    print("粒子群算法 demo：Ackley 函数最小化（dim=2）")
    print("=" * 60)
    dim = 2
    bounds = [(-5.0, 5.0)] * dim
    res = pso_minimize(ackley, dim, bounds, swarm_size=40, max_iter=200,
                       w=0.7, c1=1.5, c2=1.5, seed=42)

    print(f"理论全局最优：x* = (0, 0)，f* = 0")
    print(f"PSO 求得最优解：{np.round(res['best_x'], 4)}")
    print(f"PSO 求得最优值：f = {res['best_f']:.4f}")
    print(f"第 1 代最优值：{res['history'][0]:.4f}，"
          f"第 200 代最优值：{res['history'][-1]:.4f}")

    # 画收敛曲线
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(res['history']) + 1), res['history'], 'g-', lw=1.5)
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('全局最优目标值')
    ax.set_title('粒子群算法求解 Ackley 函数的收敛曲线')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('pso_demo.png', dpi=150)
    print("已保存收敛曲线图：pso_demo.png")
