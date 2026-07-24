# -*- coding: utf-8 -*-
"""
模块：queueing
功能：排队论 M/M/1 模型——Little 公式解析解（L、Lq、W、Wq、rho）+ 离散事件仿真对比验证
适用题型：通用（服务系统容量规划、窗口/床位/工位配置、交通拥堵建模）
依赖：numpy, pandas
用法：直接运行 `python queueing.py` 查看 demo；或 import 后调用核心函数
"""

import numpy as np
import pandas as pd


def mm1_metrics(lam, mu):
    """M/M/1 排队系统解析解（稳态指标）。

    模型假设：泊松到达（率 lam）、指数服务（率 mu）、单服务台、
    无限队长、先到先服务（FCFS）。稳态条件：lam < mu。

    参数
    ----
    lam : float
        平均到达率（顾客/单位时间）。
    mu : float
        平均服务率（顾客/单位时间）。

    返回
    ----
    result : dict
        {'rho': 服务强度（利用率）lam/mu,
         'L':  系统内平均顾客数,
         'Lq': 平均排队长（等待队列中的顾客数）,
         'W':  平均逗留时间（排队 + 服务）,
         'Wq': 平均排队等待时间}
    """
    if lam >= mu:
        raise ValueError(f'系统不稳定：到达率 lam={lam} >= 服务率 mu={mu}，'
                         '队长将趋于无穷，无稳态解。请满足 lam < mu。')

    rho = lam / mu             # 服务强度（利用率）
    L = rho / (1 - rho)        # 系统内平均顾客数 = lam/(mu-lam)
    Lq = rho ** 2 / (1 - rho)  # 平均排队长 = L - rho
    W = 1.0 / (mu - lam)       # 平均逗留时间
    Wq = rho / (mu - lam)      # 平均排队等待时间 = W - 1/mu
    return {'rho': rho, 'L': L, 'Lq': Lq, 'W': W, 'Wq': Wq}


def mm1_simulate(lam, mu, n_customers=10000, seed=None):
    """M/M/1 离散事件仿真：逐顾客模拟到达-排队-服务-离开全过程。

    仿真逻辑（逐顾客递推实现）：
      - 到达间隔 ~ Exp(lam)，服务时长 ~ Exp(mu)；
      - 顾客服务开始时刻 = max(本人到达时刻, 前一顾客离开时刻)（单服务台 FCFS，
        即 Lindley 递推）；
      - 为削弱系统从空置起步的瞬态影响，统计时舍弃前 10% 顾客（热身期）。

    参数
    ----
    lam : float
        平均到达率（顾客/单位时间）。
    mu : float
        平均服务率（顾客/单位时间）。
    n_customers : int
        仿真顾客数（默认 10000）。利用率 rho 越接近 1，队列长度波动越大，
        需要更多顾客数才能使仿真值逼近解析解。

    返回
    ----
    result : dict
        {'rho', 'L', 'Lq', 'W', 'Wq': 仿真估计值（口径同 mm1_metrics）,
         'n_customers': 仿真顾客数}
        其中 L、Lq 由面积法（Little 定律）计算：L = Σ逗留时间 / 仿真总时长。
    """
    if lam >= mu:
        raise ValueError(f'系统不稳定：到达率 lam={lam} >= 服务率 mu={mu}，'
                         '队长将趋于无穷。请满足 lam < mu。')

    if seed is not None:
        np.random.seed(seed)

    # 1) 生成到达过程：到达间隔服从指数分布，累计和为到达时刻
    inter_arrivals = np.random.exponential(1.0 / lam, n_customers)
    arrivals = np.cumsum(inter_arrivals)

    # 2) 生成每位顾客的服务时长
    services = np.random.exponential(1.0 / mu, n_customers)

    # 3) 递推服务开始与离开时刻（单服务台 FCFS 的核心差分关系）
    start = np.empty(n_customers)   # 服务开始时刻
    depart = np.empty(n_customers)  # 离开时刻
    for k in range(n_customers):
        prev_depart = depart[k - 1] if k > 0 else 0.0
        start[k] = max(arrivals[k], prev_depart)  # 到了且服务台空闲才能开始
        depart[k] = start[k] + services[k]

    # 4) 统计个体指标（舍弃前 10% 顾客，规避系统空置起步的瞬态偏差）
    n_warmup = n_customers // 10
    wait_in_queue = start[n_warmup:] - arrivals[n_warmup:]  # 排队等待时间
    dwell = depart[n_warmup:] - arrivals[n_warmup:]         # 逗留时间 = 等待 + 服务
    Wq = wait_in_queue.mean()
    W = dwell.mean()

    # 5) 系统人数指标用面积法（即 Little 定律）：L = Σ逗留 / 统计总时长
    total_time = depart[-1] - arrivals[n_warmup]
    L = dwell.sum() / total_time
    Lq = wait_in_queue.sum() / total_time

    return {'rho': lam / mu, 'L': L, 'Lq': Lq, 'W': W, 'Wq': Wq,
            'n_customers': n_customers}


def compare_analytic_vs_simulation(lam, mu, n_customers=10000, seed=None):
    """对比同一组 (lam, mu) 下的解析解与仿真解，返回对比 DataFrame。

    参数
    ----
    lam, mu : float
        到达率与服务率。
    n_customers : int
        仿真顾客数。
    seed : int or None
        随机种子（传给 mm1_simulate，保证可复现）。

    返回
    ----
    table : pandas.DataFrame
        列：指标、解析解、仿真解、相对误差(%)。
    """
    ana = mm1_metrics(lam, mu)
    sim = mm1_simulate(lam, mu, n_customers=n_customers, seed=seed)

    names = {'rho': '服务强度 rho', 'L': '系统内平均顾客数 L',
             'Lq': '平均排队长 Lq', 'W': '平均逗留时间 W',
             'Wq': '平均排队等待时间 Wq'}
    rows = []
    for key in ['rho', 'L', 'Lq', 'W', 'Wq']:
        a, s = ana[key], sim[key]
        rel_err = abs(s - a) / abs(a) * 100 if a != 0 else 0.0
        rows.append([names[key], a, s, rel_err])
    return pd.DataFrame(rows, columns=['指标', '解析解', '仿真解', '相对误差(%)'])


if __name__ == '__main__':
    print('=' * 60)
    print('M/M/1 排队模型 demo：解析解 vs 离散事件仿真')
    print('=' * 60)

    n_customers = 10000  # 按 SPEC：仿真顾客数 10000

    # ---------- 场景 A：中等负荷（rho = 0.5） ----------
    lam_a, mu_a = 6.0, 12.0
    print(f'\n【场景 A】银行网点（中等负荷）：lam = {lam_a} 人/小时，'
          f'mu = {mu_a} 人/小时，仿真顾客数 = {n_customers}')
    table_a = compare_analytic_vs_simulation(lam_a, mu_a, n_customers, seed=42)
    print(table_a.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    ana_a = mm1_metrics(lam_a, mu_a)
    print(f'  直观解读：平均排队 {ana_a["Wq"] * 60:.4f} 分钟，'
          f'从到达到离开平均 {ana_a["W"] * 60:.4f} 分钟；'
          f'柜台利用率 {ana_a["rho"] * 100:.2f}%。')

    # ---------- 场景 B：高负荷（rho ≈ 0.83） ----------
    lam_b, mu_b = 10.0, 12.0
    print(f'\n【场景 B】银行网点（高负荷）：lam = {lam_b} 人/小时，'
          f'mu = {mu_b} 人/小时，仿真顾客数 = {n_customers}')
    table_b = compare_analytic_vs_simulation(lam_b, mu_b, n_customers, seed=42)
    print(table_b.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    ana_b = mm1_metrics(lam_b, mu_b)
    print(f'  直观解读：平均排队 {ana_b["Wq"] * 60:.4f} 分钟，'
          f'从到达到离开平均 {ana_b["W"] * 60:.4f} 分钟；'
          f'柜台利用率 {ana_b["rho"] * 100:.2f}%。')

    print('\n结论：中低负荷下仿真解与解析解吻合良好；')
    print('利用率 rho 越接近 1，队列长度波动越大、相邻顾客等待时间相关性越强，')
    print('同样 10000 名顾客的仿真误差也越大——此时应增大仿真顾客数（如 10 万）再对比。')
