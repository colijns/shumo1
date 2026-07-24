# -*- coding: utf-8 -*-
"""
模块：shortest_path
功能：基于 networkx 的带权图最短路径求解（Dijkstra 单源最短路 + 全点对最短路表）与路径可视化
适用题型：通用（路网规划、物流配送、最优换乘等图论题高频）
依赖：numpy, pandas, matplotlib, networkx
用法：直接运行 `python shortest_path.py` 查看 demo；或 import 后调用核心函数
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def build_weighted_graph(edges):
    """根据边列表构造无向带权图。

    参数
    ----
    edges : list of tuple
        每个元素为 (起点, 终点, 权重)，权重须为非负数（Dijkstra 算法要求）。

    返回
    ----
    G : networkx.Graph
        构造好的无向带权图，边权存放在 'weight' 属性中。
    """
    G = nx.Graph()
    # 批量添加带权边，weight 作为 Dijkstra 的距离度量
    G.add_weighted_edges_from(edges)
    return G


def single_source_shortest(G, source):
    """Dijkstra 单源最短路：求 source 到所有节点的最短距离与具体路径。

    参数
    ----
    G : networkx.Graph
        带权图（边权为 'weight'）。
    source : 节点标签
        源点。

    返回
    ----
    result : dict
        {'distances': {节点: 最短距离}, 'paths': {节点: [路径节点列表]}}
    """
    # 距离字典与路径字典，均为 Dijkstra 算法
    distances = nx.single_source_dijkstra_path_length(G, source, weight='weight')
    paths = nx.single_source_dijkstra_path(G, source, weight='weight')
    return {'distances': distances, 'paths': paths}


def all_pairs_shortest(G):
    """全点对最短路：计算任意两节点间的最短距离，返回对称距离表。

    参数
    ----
    G : networkx.Graph
        带权图。

    返回
    ----
    dist_df : pandas.DataFrame
        行、列均为节点标签的最短距离矩阵（对角线为 0）。
    """
    nodes = list(G.nodes())
    # dict_of_dicts：dist[u][v] 为 u 到 v 的最短距离
    dist_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    dist_df = pd.DataFrame(dist_dict).reindex(index=nodes, columns=nodes)
    return dist_df


def plot_shortest_path(G, path, title='最短路径示意图'):
    """绘制图结构并高亮指定路径。

    参数
    ----
    G : networkx.Graph
        带权图。
    path : list
        需要高亮的路径节点序列（如 dijkstra 求得的某条最短路）。
    title : str
        图标题。

    返回
    ----
    fig : matplotlib.figure.Figure
        画好的图对象。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # 固定随机布局种子，保证每次运行图结构一致
    pos = nx.spring_layout(G, seed=42)

    # 先画全部节点与边（浅色打底）
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    # 标注边权
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)

    # 将路径拆成相邻边并红色加粗高亮
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=800, ax=ax)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print('=' * 50)
    print('最短路径 demo：带权路网 Dijkstra 求解')
    print('=' * 50)

    # 构造一个 7 节点的带权路网（模拟城市间公路里程，单位：百公里）
    edges = [
        ('A', 'B', 4), ('A', 'C', 2),
        ('B', 'C', 1), ('B', 'D', 5),
        ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6),
        ('E', 'F', 3), ('F', 'G', 4),
        ('E', 'G', 7),
    ]
    G = build_weighted_graph(edges)
    print(f'图规模：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边')

    # 1) 单源最短路：以 A 为源点
    source = 'A'
    res = single_source_shortest(G, source)
    print(f'\n【1】以 {source} 为源点的单源最短路：')
    for node in sorted(res['distances']):
        path_str = ' -> '.join(res['paths'][node])
        print(f'  {source} -> {node}: 最短距离 = {res["distances"][node]:.4f}, 路径: {path_str}')

    # 2) 全点对最短路表
    dist_df = all_pairs_shortest(G)
    print('\n【2】全点对最短距离表（单位：百公里）：')
    print(dist_df.to_string(float_format=lambda v: f'{v:.4f}'))

    # 3) 可视化 A 到 G 的最短路径
    target_path = res['paths']['G']
    fig = plot_shortest_path(G, target_path, title=f'{source} 到 G 的最短路径（红色高亮）')
    fig.savefig('shortest_path_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\n路径图已保存为 shortest_path_demo.png。')
