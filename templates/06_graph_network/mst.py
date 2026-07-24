# -*- coding: utf-8 -*-
"""
模块：mst
功能：最小生成树（Minimum Spanning Tree）求解——城市联网最小造价问题，输出边集与总造价并画图
适用题型：通用（电网/路网/通信网铺设等连通成本最小化题型）
依赖：numpy, matplotlib, networkx
用法：直接运行 `python mst.py` 查看 demo；或 import 后调用核心函数
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def build_cost_graph(cities, edges):
    """根据城市列表与造价边列表构造无向带权图。

    参数
    ----
    cities : list
        城市（节点）名称列表。
    edges : list of tuple
        每个元素为 (城市1, 城市2, 造价)，造价即边权 'weight'。

    返回
    ----
    G : networkx.Graph
        构造好的无向带权图。
    """
    G = nx.Graph()
    G.add_nodes_from(cities)
    G.add_weighted_edges_from(edges)
    return G


def solve_mst(G, algorithm='kruskal'):
    """求最小生成树，返回生成树边集与总权重。

    参数
    ----
    G : networkx.Graph
        带权连通图（边权为 'weight'）。
    algorithm : str
        求解算法，可选 'kruskal'（默认）、'prim'、'boruvka'。

    返回
    ----
    result : dict
        {'edges': [(u, v, weight), ...] 生成树边集（按权重升序）,
         'total_weight': float 总权重（总造价）,
         'mst_graph': networkx.Graph 生成树子图}
    """
    # networkx 内置最小生成树算法（默认 Kruskal）
    mst = nx.minimum_spanning_tree(G, weight='weight', algorithm=algorithm)

    # 提取边集并按权重升序排列，便于阅读
    edges = [(u, v, d['weight']) for u, v, d in mst.edges(data=True)]
    edges.sort(key=lambda e: e[2])
    total_weight = sum(w for _, _, w in edges)

    return {'edges': edges, 'total_weight': total_weight, 'mst_graph': mst}


def plot_mst(G, mst, title='最小生成树示意图'):
    """绘制原始网络，并红色加粗高亮最小生成树。

    参数
    ----
    G : networkx.Graph
        原始带权图。
    mst : networkx.Graph
        最小生成树子图。
    title : str
        图标题。

    返回
    ----
    fig : matplotlib.figure.Figure
        画好的图对象。
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # 固定布局种子，保证结果可复现
    pos = nx.spring_layout(G, seed=42)

    # 打底：全部节点与边（灰色细边）
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    # 标注原图全部边权（造价）
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)

    # 高亮：MST 边红色加粗
    nx.draw_networkx_edges(G, pos, edgelist=list(mst.edges()),
                           edge_color='red', width=3, ax=ax)

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    print('=' * 50)
    print('最小生成树 demo：城市联网最小造价问题')
    print('=' * 50)

    # 7 座城市，任意两城间铺设光缆的造价（单位：万元），求连通全网的最小总造价
    cities = ['北京', '天津', '石家庄', '济南', '郑州', '太原', '呼和浩特']
    edges = [
        ('北京', '天津', 12), ('北京', '石家庄', 28), ('北京', '太原', 50),
        ('北京', '呼和浩特', 46), ('天津', '济南', 30), ('石家庄', '济南', 26),
        ('石家庄', '郑州', 41), ('石家庄', '太原', 21), ('济南', '郑州', 36),
        ('郑州', '太原', 38), ('太原', '呼和浩特', 42), ('天津', '石家庄', 27),
    ]
    G = build_cost_graph(cities, edges)
    print(f'网络规模：{G.number_of_nodes()} 座城市，{G.number_of_edges()} 条候选线路')

    # 求解最小生成树（Kruskal）
    res = solve_mst(G, algorithm='kruskal')
    print('\n【1】最小生成树边集（按造价升序）：')
    for u, v, w in res['edges']:
        print(f'  {u} -- {v}: 造价 {w:.4f} 万元')
    print(f'\n【2】连通全部城市的最小总造价：{res["total_weight"]:.4f} 万元')
    print(f'  生成树边数 = {len(res["edges"])}（应等于城市数 - 1 = {len(cities) - 1}）')

    # 画图：灰色为候选线路，红色为选中的 MST 线路
    fig = plot_mst(G, res['mst_graph'], title='城市联网最小生成树（红色为选中线路）')
    fig.savefig('mst_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('\n最小生成树图已保存为 mst_demo.png。')
