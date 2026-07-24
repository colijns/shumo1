# -*- coding: utf-8 -*-
"""
模块：max_flow
功能：最大流问题求解——networkx maximum_flow，输出最大流量与各边流量分配并画图
适用题型：通用（输油管网、交通运力、供应链配送等网络流题型）
依赖：numpy, matplotlib, networkx
用法：直接运行 `python max_flow.py` 查看 demo；或 import 后调用核心函数
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def build_flow_network(edges):
    """根据边列表构造有向容量网络。

    参数
    ----
    edges : list of tuple
        每个元素为 (起点, 终点, 容量)，容量存放在 'capacity' 属性中。

    返回
    ----
    G : networkx.DiGraph
        构造好的有向容量网络。
    """
    G = nx.DiGraph()
    # 逐条添加边并设置容量属性
    for u, v, cap in edges:
        G.add_edge(u, v, capacity=cap)
    return G


def solve_max_flow(G, source, sink):
    """求源点到汇点的最大流及各边流量分配。

    参数
    ----
    G : networkx.DiGraph
        有向容量网络（边属性 'capacity'）。
    source : 节点标签
        源点。
    sink : 节点标签
        汇点。

    返回
    ----
    result : dict
        {'max_flow_value': float 最大流量,
         'edge_flows': {(u, v): (流量, 容量)} 各边流量分配,
         'min_cut': (reachable, non_reachable) 最小割两侧节点集合}
    """
    # Ford-Fulkerson 类算法求最大流；flow_dict[u][v] 为边 (u,v) 上的流量
    flow_value, flow_dict = nx.maximum_flow(G, source, sink)

    # 整理成 {(u, v): (流量, 容量)} 形式，便于打印与后续分析
    edge_flows = {}
    for u, nbrs in flow_dict.items():
        for v, f in nbrs.items():
            edge_flows[(u, v)] = (f, G[u][v]['capacity'])

    # 最小割（最大流最小割定理：割容量 = 最大流量）
    cut_value, (reachable, non_reachable) = nx.minimum_cut(G, source, sink)

    return {'max_flow_value': flow_value, 'edge_flows': edge_flows,
            'min_cut': (reachable, non_reachable)}


def plot_flow_network(G, edge_flows, source, sink, title='最大流网络示意图'):
    """绘制容量网络，边标签显示「流量/容量」，并高亮源点与汇点。

    参数
    ----
    G : networkx.DiGraph
        有向容量网络。
    edge_flows : dict
        {(u, v): (流量, 容量)} 各边流量分配。
    source, sink : 节点标签
        源点与汇点（着色区分）。
    title : str
        图标题。

    返回
    ----
    fig : matplotlib.figure.Figure
        画好的图对象。
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    # 固定布局种子，保证结果可复现
    pos = nx.spring_layout(G, seed=42)

    # 节点着色：源点绿色、汇点红色、中间节点浅蓝
    node_colors = []
    for n in G.nodes():
        if n == source:
            node_colors.append('lightgreen')
        elif n == sink:
            node_colors.append('salmon')
        else:
            node_colors.append('lightblue')
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

    # 有向边（带箭头）
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2,
                           arrows=True, arrowsize=20, ax=ax)

    # 边标签：流量/容量（满载边红色显示，便于识别瓶颈）
    edge_labels = {(u, v): f'{f}/{c}' for (u, v), (f, c) in edge_flows.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    saturated = [(u, v) for (u, v), (f, c) in edge_flows.items() if f == c and c > 0]
    nx.draw_networkx_edges(G, pos, edgelist=saturated, edge_color='red',
                           width=3, arrows=True, arrowsize=20, ax=ax)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print('=' * 50)
    print('最大流 demo：输油管网最大输送能力')
    print('=' * 50)

    # 经典输油管网：S 为油库（源点），T 为炼厂（汇点），中间为中转站
    # 边 (u, v, 容量)，容量单位：万吨/日
    edges = [
        ('S', 'V1', 10), ('S', 'V2', 8),
        ('V1', 'V2', 3), ('V1', 'V3', 6),
        ('V2', 'V4', 7), ('V3', 'V4', 4),
        ('V3', 'T', 5), ('V4', 'T', 12),
        ('V2', 'V3', 2),
    ]
    G = build_flow_network(edges)
    source, sink = 'S', 'T'
    print(f'网络规模：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条管道，'
          f'源点 {source}，汇点 {sink}')

    # 求最大流
    res = solve_max_flow(G, source, sink)
    print(f'\n【1】最大流量：{res["max_flow_value"]:.4f} 万吨/日')

    print('\n【2】各管道流量分配（流量/容量，万吨/日）：')
    for (u, v), (f, c) in sorted(res['edge_flows'].items()):
        mark = '  <-- 满载（瓶颈）' if f == c and c > 0 else ''
        print(f'  {u} -> {v}: {f:.4f} / {c:.4f}{mark}')

    reachable, non_reachable = res['min_cut']
    print(f'\n【3】最小割：源侧节点 {sorted(reachable)} | 汇侧节点 {sorted(non_reachable)}')
    print('  （由最大流最小割定理，最小割容量 = 最大流量）')

    # 画图：红色边为满载管道（瓶颈）
    fig = plot_flow_network(G, res['edge_flows'], source, sink,
                            title=f'最大流网络（最大流量 = {res["max_flow_value"]:.0f}，红色为满载管道）')
    fig.savefig('max_flow_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\n最大流网络图已保存为 max_flow_demo.png。')
