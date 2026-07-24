# -*- coding: utf-8 -*-
"""
模块：genetic_algorithm
功能：手写遗传算法（GA），实数编码、锦标赛选择、算术交叉、高斯变异，求解连续函数最小化
适用题型：通用高频（复杂非线性全局优化、无梯度问题、组合优化初解等）
依赖：numpy, matplotlib（不依赖任何第三方优化库）
用法：直接运行 `python genetic_algorithm.py` 查看 demo；或 import 后调用 ga_minimize
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def ga_minimize(func, dim, bounds, pop_size=50, max_gen=200, pc=0.8, pm=0.1, seed=None):
    """
    用遗传算法求 func 的最小值（实数编码）。

    参数：
        func     : 目标函数，接收 shape (dim,) 的向量，返回标量
        dim      : 决策变量维数
        bounds   : 变量边界，shape (dim, 2)，每行为 (下界, 上界)
        pop_size : 种群规模
        max_gen  : 最大进化代数
        pc       : 交叉概率（算术交叉）
        pm       : 变异概率（高斯变异）
        seed     : 随机种子，int 或 None（None 表示不固定种子）

    返回：
        dict，键为：
            'best_x'  : 最优个体（numpy 数组，shape (dim,)）
            'best_f'  : 最优目标值（float）
            'history' : 每代最优目标值列表（长度 max_gen）
    """
    if seed is not None:
        np.random.seed(seed)
    bounds = np.asarray(bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb

    # 1. 初始化种群：边界内均匀随机
    pop = lb + np.random.rand(pop_size, dim) * span
    fit = np.array([func(ind) for ind in pop])

    history = []
    for gen in range(max_gen):
        # 记录当前最优
        best_idx = np.argmin(fit)
        history.append(float(fit[best_idx]))
        best_ind = pop[best_idx].copy()

        # 2. 锦标赛选择（规模为 3）：随机抽 3 个个体，取适应度最小者
        new_pop = []
        for _ in range(pop_size):
            cand = np.random.choice(pop_size, size=3, replace=False)
            winner = cand[np.argmin(fit[cand])]
            new_pop.append(pop[winner].copy())
        new_pop = np.array(new_pop)

        # 3. 算术交叉：对每对相邻个体以概率 pc 做线性组合
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < pc:
                alpha = np.random.rand(dim)          # 各维度独立交叉系数
                p1, p2 = new_pop[i], new_pop[i + 1]
                new_pop[i] = alpha * p1 + (1 - alpha) * p2
                new_pop[i + 1] = alpha * p2 + (1 - alpha) * p1

        # 4. 高斯变异：以概率 pm 对基因加高斯扰动（步长取边界跨度的 5%）
        for i in range(pop_size):
            if np.random.rand() < pm:
                noise = np.random.randn(dim) * 0.05 * span
                new_pop[i] = new_pop[i] + noise

        # 越界截断回边界
        new_pop = np.clip(new_pop, lb, ub)

        # 5. 精英保留：用上一代最优个体替换新种群中随机一个位置，防止退化
        keep = np.random.randint(pop_size)
        new_pop[keep] = best_ind

        pop = new_pop
        fit = np.array([func(ind) for ind in pop])

    # 最终最优
    best_idx = np.argmin(fit)
    return {
        'best_x': pop[best_idx].copy(),
        'best_f': float(fit[best_idx]),
        'history': history,
    }


def rastrigin(x):
    """Rastrigin 测试函数：多峰，全局最小值在原点，f(0)=0。"""
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    print("=" * 60)
    print("遗传算法 demo：Rastrigin 函数最小化（dim=2）")
    print("=" * 60)
    dim = 2
    bounds = [(-5.12, 5.12)] * dim
    res = ga_minimize(rastrigin, dim, bounds, pop_size=50, max_gen=200, pc=0.8, pm=0.1, seed=42)

    print(f"理论全局最优：x* = (0, 0)，f* = 0")
    print(f"GA 求得最优解：{np.round(res['best_x'], 4)}")
    print(f"GA 求得最优值：f = {res['best_f']:.4f}")
    print(f"第 1 代最优值：{res['history'][0]:.4f}，"
          f"第 200 代最优值：{res['history'][-1]:.4f}")

    # 画收敛曲线
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(res['history']) + 1), res['history'], 'b-', lw=1.5)
    ax.set_xlabel('进化代数')
    ax.set_ylabel('当代最优目标值')
    ax.set_title('遗传算法求解 Rastrigin 函数的收敛曲线')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ga_demo.png', dpi=150)
    print("已保存收敛曲线图：ga_demo.png")
