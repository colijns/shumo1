# -*- coding: utf-8 -*-
"""
模块：monte_carlo
功能：蒙特卡洛模拟——随机投点估算圆周率 π、定积分数值估计、报童模型库存利润仿真
适用题型：通用高频（2024 国赛 B 题主流方法；概率模拟、复杂系统仿真类赛题）
依赖：numpy, matplotlib
用法：直接运行 `python monte_carlo.py` 查看 demo；或 import 后调用
      estimate_pi / monte_carlo_integral / simulate_inventory
"""
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def estimate_pi(n=100000, seed=None):
    """蒙特卡洛随机投点法估算圆周率 π

    原理：在单位正方形 [0,1]×[0,1] 内均匀投点，落入单位圆四分之一
    （x²+y²<=1）的概率为 π/4，故 π ≈ 4 × 命中比例

    参数:
        n: int, 投点总数，默认 100000
        seed: int 或 None, 随机种子；传入整数可复现，None 则每次不同
    返回:
        float: π 的估计值
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    inside = (x ** 2 + y ** 2) <= 1.0  # 是否落入四分之一圆内
    return 4.0 * np.mean(inside)


def monte_carlo_integral(func, a, b, n=100000, seed=None):
    """蒙特卡洛均值法估计一维定积分 ∫_a^b func(x) dx

    原理：∫_a^b f(x)dx = (b-a) × E[f(X)]，X 服从 [a,b] 上均匀分布，
    用样本均值近似期望：积分 ≈ (b-a) × mean(f(x_i))

    参数:
        func: callable, 被积函数（需支持 numpy 数组向量化运算）
        a, b: float, 积分下、上限
        n: int, 抽样点数，默认 100000
        seed: int 或 None, 随机种子；传入整数可复现，None 则每次不同
    返回:
        float: 定积分估计值
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(a, b, n)
    return float((b - a) * np.mean(func(x)))


def simulate_inventory(days=365, demand_lambda=50, stock=55,
                       price=10.0, cost=6.0, salvage=3.0, seed=None):
    """报童模型库存仿真：固定订货量策略下统计期望日利润

    模型设定（报童问题简化版）：
        每日需求量 D ~ Poisson(demand_lambda)；
        每天开店前备货 stock 件，售价 price，进价 cost，
        当日未售出部分以 salvage 清仓价处理；
        日利润 = price×售出量 + salvage×剩余量 - cost×备货量

    参数:
        days: int, 模拟天数，默认 365
        demand_lambda: float, 日需求量的泊松分布参数（期望需求），默认 50
        stock: int, 每日固定备货量，默认 55
        price: float, 零售价，默认 10.0
        cost: float, 进货成本，默认 6.0
        salvage: float, 清仓处理价，默认 3.0
        seed: int 或 None, 随机种子；传入整数可复现，None 则每次不同
    返回:
        dict:
            'daily_profit'  每日利润序列（numpy 数组，长度 days）
            'mean_profit'   期望（平均）日利润
            'std_profit'    日利润标准差（风险度量）
            'sold_out_rate' 售罄（缺货）天数占比
            'params'        仿真参数快照
    """
    if seed is not None:
        np.random.seed(seed)
    demand = np.random.poisson(demand_lambda, days)  # 每日需求量
    sold = np.minimum(demand, stock)                 # 实际售出量（受库存限制）
    leftover = np.maximum(stock - demand, 0)         # 剩余清仓量
    daily_profit = price * sold + salvage * leftover - cost * stock
    return {
        'daily_profit': daily_profit,
        'mean_profit': float(np.mean(daily_profit)),
        'std_profit': float(np.std(daily_profit, ddof=1)),
        'sold_out_rate': float(np.mean(demand > stock)),
        'params': {
            'days': days, 'demand_lambda': demand_lambda, 'stock': stock,
            'price': price, 'cost': cost, 'salvage': salvage,
        },
    }


if __name__ == '__main__':
    # ---------------- demo 1：蒙特卡洛估算 π 并画收敛图 ----------------
    print('=' * 60)
    print('蒙特卡洛模拟 demo')
    print('=' * 60)
    n = 100000
    pi_hat = estimate_pi(n, seed=42)
    print(f'\n1. 估算圆周率：投点 {n} 次，π ≈ {pi_hat:.4f}（真值 3.1416，'
          f'误差 {abs(pi_hat - np.pi):.4f}）')

    # π 收敛图：估计值随投点数的变化（重新采样并计算累计估计）
    np.random.seed(42)
    xs = np.random.uniform(0, 1, n)
    ys = np.random.uniform(0, 1, n)
    hits = np.cumsum((xs ** 2 + ys ** 2) <= 1.0)  # 累计命中数
    pi_running = 4.0 * hits / np.arange(1, n + 1)  # 累计估计值
    plt.figure(figsize=(8, 5))
    plt.plot(pi_running, linewidth=0.8, label='π 的累计估计值')
    plt.axhline(np.pi, color='red', linestyle='--', label='π 真值 3.1416')
    plt.xscale('log')
    plt.xlabel('投点数（对数刻度）')
    plt.ylabel('π 估计值')
    plt.title('蒙特卡洛估算 π 的收敛过程')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('monte_carlo_demo1.png', dpi=150, bbox_inches='tight')
    print('  -> 已保存 monte_carlo_demo1.png')

    # ---------------- demo 2：蒙特卡洛估计定积分 ----------------
    # 例：∫_0^2 x²·exp(-x) dx，解析解 = 2 - 10e^(-2) ≈ 0.6466
    integral_hat = monte_carlo_integral(lambda x: x ** 2 * np.exp(-x), 0, 2, seed=42)
    exact = 2 - 10 * np.exp(-2)
    print(f'\n2. 定积分 int_0^2 x^2*e^(-x)dx：蒙特卡洛估计 = {integral_hat:.4f}，'
          f'解析解 = {exact:.4f}，误差 {abs(integral_hat - exact):.4f}')

    # ---------------- demo 3：报童模型库存仿真 ----------------
    print('\n3. 报童模型库存仿真（365 天）：')
    result = simulate_inventory(days=365, seed=42)
    print(f"   期望日利润 = {result['mean_profit']:.4f} 元，"
          f"利润标准差 = {result['std_profit']:.4f} 元，"
          f"售罄率 = {result['sold_out_rate']:.4f}")

    # 灵敏度分析：比较不同备货量下的期望利润，寻找最优备货量
    # 每轮 seed=42 使用相同需求序列——有意的对照实验（只变 stock，需求固定）
    stocks = range(30, 81, 5)
    mean_profits = []
    for s in stocks:
        r = simulate_inventory(days=365, stock=s, seed=42)
        mean_profits.append(r['mean_profit'])
    best_stock = list(stocks)[int(np.argmax(mean_profits))]
    print(f'   灵敏度分析：备货量 {list(stocks)} 中，'
          f'期望利润最大的最优备货量 = {best_stock} 件')

    plt.figure(figsize=(8, 5))
    plt.plot(list(stocks), mean_profits, 'o-', color='darkorange')
    plt.axvline(best_stock, color='red', linestyle='--',
                label=f'最优备货量 {best_stock} 件')
    plt.xlabel('每日备货量（件）')
    plt.ylabel('期望日利润（元）')
    plt.title('报童模型：备货量对期望利润的影响')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('monte_carlo_demo2.png', dpi=150, bbox_inches='tight')
    print('  -> 已保存 monte_carlo_demo2.png')
    print('demo 运行结束。')
