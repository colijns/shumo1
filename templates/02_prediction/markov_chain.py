# -*- coding: utf-8 -*-
"""
模块：markov_chain
功能：马尔可夫链预测——n 步转移概率、状态分布演化与稳态（极限分布）分析
适用题型：通用高频（市场占有率、状态转移、期望收益类问题，国赛C题/华数杯常用）
依赖：numpy, matplotlib
用法：直接运行 `python markov_chain.py` 查看 demo；或 import 后使用 MarkovChain 类
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


class MarkovChain:
    """离散时间马尔可夫链

    属性:
        P: ndarray, 一步转移概率矩阵（行和为 1，shape (n_states, n_states)）
        states: list, 状态名称
    """

    def __init__(self, P, states=None):
        """构造马尔可夫链

        参数:
            P: array_like, 一步转移概率矩阵，P[i, j] 表示从状态 i 转移到状态 j 的概率
            states: list 或 None, 状态名称，缺省为 ['S0', 'S1', ...]
        """
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError('转移矩阵必须为方阵')
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError('转移矩阵每行之和必须等于 1')
        if np.any(P < 0):
            raise ValueError('转移概率必须非负')
        self.P = P
        self.n_states = P.shape[0]
        self.states = list(states) if states is not None \
            else [f'S{i}' for i in range(self.n_states)]

    def n_step(self, n):
        """计算 n 步转移概率矩阵 P^n

        参数:
            n: int, 步数（n >= 1）
        返回:
            numpy.ndarray: n 步转移概率矩阵，[i, j] 为从状态 i 经 n 步到状态 j 的概率
        """
        if n < 1:
            raise ValueError('步数 n 必须 >= 1')
        return np.linalg.matrix_power(self.P, n)

    def predict_distribution(self, s0, n):
        """由初始状态分布 s0 预测 n 步后的状态分布：s_n = s0 @ P^n

        参数:
            s0: array_like, 初始状态分布（行向量，和为 1）
            n: int, 步数
        返回:
            numpy.ndarray: n 步后的状态分布
        """
        s0 = np.asarray(s0, dtype=float).ravel()
        return s0 @ self.n_step(n)

    def steady_state(self):
        """求解稳态分布（极限分布）π，满足 π = πP 且 sum(π) = 1

        做法：解线性方程组 (P^T - I)π = 0，并把其中一个方程替换为 sum(π)=1

        返回:
            numpy.ndarray: 稳态分布向量（和为 1）
        """
        n = self.n_states
        A = self.P.T - np.eye(n)   # π = πP 等价于 (P^T - I)π = 0
        A[-1, :] = 1.0             # 用归一化条件替换最后一个方程
        b = np.zeros(n)
        b[-1] = 1.0
        pi = np.linalg.solve(A, b)
        return pi


if __name__ == '__main__':
    # ---------------- demo：三家超市市场占有率预测与稳态分析 ----------------
    # 状态：甲、乙、丙三家超市；P[i,j] 为顾客本月在 i 超市、下月转向 j 超市的概率
    states = ['甲超市', '乙超市', '丙超市']
    P = np.array([
        [0.7, 0.2, 0.1],   # 甲超市顾客的留存与流失
        [0.3, 0.5, 0.2],   # 乙超市顾客的留存与流失
        [0.2, 0.2, 0.6],   # 丙超市顾客的留存与流失
    ])
    s0 = np.array([0.40, 0.35, 0.25])  # 当前市场占有率

    print('=' * 60)
    print('马尔可夫链 demo：三家超市市场占有率预测与稳态分析')
    print('=' * 60)
    mc = MarkovChain(P, states=states)

    # 第一步：n 步转移矩阵示例
    n = 3
    Pn = mc.n_step(n)
    print(f'{n} 步转移概率矩阵 P^{n}:')
    print(np.round(Pn, 4))

    # 第二步：预测未来 1~6 个月的市场占有率
    print('\n未来各月市场占有率预测：')
    print('月份   ' + '   '.join(states))
    history = [s0]
    for month in range(1, 7):
        s_n = mc.predict_distribution(s0, month)
        history.append(s_n)
        print(f'{month:>3d}   ' + '   '.join(f'{v:.4f}' for v in s_n))

    # 第三步：稳态（长期）市场占有率
    pi = mc.steady_state()
    print('\n稳态分布（长期市场占有率）:')
    for name, v in zip(states, pi):
        print(f'  {name}: {v:.4f}')

    # 画市场占有率随月份演化曲线
    history = np.array(history)  # 第 0 行为初始分布
    plt.figure(figsize=(8, 5))
    for j, name in enumerate(states):
        plt.plot(range(0, 7), history[:, j], 'o-', label=name)
        plt.axhline(pi[j], linestyle='--', alpha=0.4)
    plt.xlabel('月份')
    plt.ylabel('市场占有率')
    plt.title('马尔可夫链市场占有率演化（虚线为稳态水平）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('markov_chain_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('demo 运行结束。')
