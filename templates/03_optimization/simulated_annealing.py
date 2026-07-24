# -*- coding: utf-8 -*-
"""
模块：simulated_annealing
功能：手写模拟退火算法（SA），含连续函数最小化与旅行商问题（TSP）两个接口
适用题型：通用高频（TSP/路径规划、组合优化、全局优化跳出局部最优）
依赖：numpy, matplotlib（不依赖任何第三方优化库）
用法：直接运行 `python simulated_annealing.py` 查看 demo；或 import 后调用 sa_minimize / sa_tsp
"""

import numpy as np

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def sa_minimize(func, x0, bounds, T0=100.0, alpha=0.95, T_min=1e-3, iter_per_T=100, seed=None):
    """
    用模拟退火求连续函数 func 的最小值。

    参数：
        func       : 目标函数，接收 numpy 向量 x，返回标量
        x0         : 初始点，shape (dim,)
        bounds     : 变量边界，shape (dim, 2)
        T0         : 初始温度
        alpha      : 降温系数（0 < alpha < 1）
        T_min      : 终止温度
        iter_per_T : 每个温度下的内层迭代次数
        seed       : 随机种子，int 或 None（None 表示不固定种子）

    返回：
        dict，键为：
            'best_x'  : 最优解向量
            'best_f'  : 最优目标值
            'history' : 每个温度阶段的最优目标值列表
    """
    if seed is not None:
        np.random.seed(seed)
    bounds = np.asarray(bounds, dtype=float)
    lb, ub = bounds[:, 0], bounds[:, 1]
    span = ub - lb

    x = np.asarray(x0, dtype=float).copy()
    f = float(func(x))
    best_x, best_f = x.copy(), f

    T = T0
    history = []
    while T > T_min:
        for _ in range(iter_per_T):
            # 邻域扰动：高斯步长随温度缩小（高温大步探索，低温小步开发）
            step = np.random.randn(len(x)) * 0.1 * span * (T / T0)
            x_new = np.clip(x + step, lb, ub)
            f_new = float(func(x_new))
            delta = f_new - f
            # Metropolis 准则：更优必接受，变差以概率 exp(-delta/T) 接受
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                x, f = x_new, f_new
                if f < best_f:
                    best_x, best_f = x.copy(), f
        history.append(best_f)
        T *= alpha  # 几何降温

    return {'best_x': best_x, 'best_f': best_f, 'history': history}


def _tour_distance(route, dist_mat):
    """计算回路总长度（含返回起点）。"""
    idx = np.asarray(route)
    nxt = np.roll(idx, -1)
    return float(dist_mat[idx, nxt].sum())


def sa_tsp(coords, T0=1000, alpha=0.98, seed=None):
    """
    用模拟退火求解旅行商问题（TSP）：求经过所有城市并回到起点的最短回路。

    参数：
        coords : 城市坐标，shape (n, 2)
        T0     : 初始温度
        alpha  : 降温系数
        seed   : 随机种子，int 或 None（None 表示不固定种子）

    返回：
        dict，键为：
            'route'    : 最优访问顺序（城市下标列表，首尾不重复起点）
            'distance' : 最短回路长度
            'history'  : 每个温度阶段的最短回路长度列表
    """
    if seed is not None:
        np.random.seed(seed)
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    # 预计算距离矩阵
    diff = coords[:, None, :] - coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(axis=-1))

    # 初始解：随机排列
    route = list(np.random.permutation(n))
    cur_d = _tour_distance(route, dist_mat)
    best_route, best_d = route.copy(), cur_d

    T = float(T0)
    T_min = 1e-3
    history = []
    while T > T_min:
        # 每个温度下内层迭代 n*10 次
        for _ in range(n * 10):
            # 邻域操作：随机选择 2-opt（逆转一段子路径）或交换两个城市
            new_route = route.copy()
            if np.random.rand() < 0.5:
                i, j = sorted(np.random.choice(n, 2, replace=False))
                new_route[i:j + 1] = new_route[i:j + 1][::-1]   # 2-opt 逆转
            else:
                i, j = np.random.choice(n, 2, replace=False)
                new_route[i], new_route[j] = new_route[j], new_route[i]
            new_d = _tour_distance(new_route, dist_mat)
            delta = new_d - cur_d
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                route, cur_d = new_route, new_d
                if cur_d < best_d:
                    best_route, best_d = route.copy(), cur_d
        history.append(best_d)
        T *= alpha

    # 转为普通 int 列表，避免打印时出现 np.int64 包装
    return {'route': [int(i) for i in best_route],
            'distance': best_d, 'history': history}


def rastrigin(x):
    """Rastrigin 测试函数：全局最小值在原点，f(0)=0。"""
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


if __name__ == "__main__":
    # ---------- demo1：连续函数最小化 ----------
    print("=" * 60)
    print("模拟退火 demo1：Rastrigin 函数最小化（dim=2）")
    print("=" * 60)
    # Rastrigin 多峰，降温稍慢、内层迭代稍多，利于跳出局部最优
    res1 = sa_minimize(rastrigin, x0=[3.0, -4.0], bounds=[(-5.12, 5.12)] * 2,
                       T0=100.0, alpha=0.97, T_min=1e-3, iter_per_T=150, seed=42)
    print(f"理论全局最优：x* = (0, 0)，f* = 0")
    print(f"SA 求得最优解：{np.round(res1['best_x'], 4)}")
    print(f"SA 求得最优值：f = {res1['best_f']:.4f}")

    # ---------- demo2：TSP（20 个随机城市） ----------
    print("-" * 60)
    print("模拟退火 demo2：TSP 最短回路（20 个随机城市）")
    print("-" * 60)
    np.random.seed(42)                        # 重置种子，使 TSP 结果独立稳定
    coords = np.random.rand(20, 2) * 100      # 20 个城市，坐标在 [0,100]^2
    res2 = sa_tsp(coords, T0=1000, alpha=0.98, seed=42)
    print(f"最短回路长度：{res2['distance']:.4f}")
    print(f"访问顺序（城市编号）：{res2['route']}")

    # 可视化：左图画 TSP 路线图，右图画收敛曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    route = res2['route'] + [res2['route'][0]]      # 闭合回路
    path = coords[route]
    axes[0].plot(path[:, 0], path[:, 1], 'o-', color='steelblue',
                 markersize=6, markerfacecolor='red')
    for i, (cx, cy) in enumerate(coords):
        axes[0].annotate(str(i), (cx, cy), textcoords='offset points',
                         xytext=(4, 4), fontsize=8)
    axes[0].set_title(f'TSP 最优回路（总长 {res2["distance"]:.4f}）')
    axes[0].set_xlabel('X 坐标')
    axes[0].set_ylabel('Y 坐标')

    axes[1].plot(range(1, len(res2['history']) + 1), res2['history'],
                 'r-', lw=1.5)
    axes[1].set_xlabel('降温阶段')
    axes[1].set_ylabel('当前最短回路长度')
    axes[1].set_title('模拟退火求解 TSP 的收敛曲线')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('sa_tsp_demo.png', dpi=150)
    print("已保存路线图与收敛曲线：sa_tsp_demo.png")
