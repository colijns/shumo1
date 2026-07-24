# -*- coding: utf-8 -*-
"""
模块：hierarchical
功能：层次聚类（系统聚类），基于 scipy.cluster.hierarchy，含树状图绘制
适用题型：通用高频（样本谱系分类、指标归类、聚类数未知的探索性分析）
依赖：numpy, scipy, matplotlib
用法：直接运行 `python hierarchical.py` 查看 demo；或 import 后调用 hierarchical_cluster / plot_dendrogram
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def hierarchical_cluster(X, n_clusters=3, method='ward'):
    """
    层次聚类：先自底向上凝聚，再按簇数截取平面簇标签。

    参数：
        X          : 样本数据，shape (n_samples, n_features)
        n_clusters : 期望的簇数
        method     :  linkage 距离计算方式，常用 'ward'（离差平方和）、
                      'average'（类平均）、'complete'（最长距离）、'single'（最短距离）

    返回：
        labels : 聚类标签，shape (n_samples,)，取值 1..n_clusters（fcluster 约定从 1 开始）
    """
    X = np.asarray(X, dtype=float)
    Z = linkage(X, method=method)                      # 凝聚过程，得到链接矩阵
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')  # 按最大簇数截取
    return labels


def plot_dendrogram(X, method='ward', save_path='hierarchical_dendrogram.png'):
    """
    绘制层次聚类树状图（谱系图）。

    参数：
        X         : 样本数据，shape (n_samples, n_features)
        method    : linkage 距离计算方式
        save_path : 图片保存路径

    返回：
        Z : scipy 链接矩阵，可用于进一步分析
    """
    X = np.asarray(X, dtype=float)
    Z = linkage(X, method=method)
    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8,
               color_threshold=None)
    ax.set_title(f'层次聚类树状图（method={method}）')
    ax.set_xlabel('样本编号')
    ax.set_ylabel('簇间距离')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    return Z


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现

    print("=" * 60)
    print("层次聚类 demo：三维随机数据（3 个真实簇）")
    print("=" * 60)
    # 构造 3 个中心的随机数据
    centers = np.array([[0, 0, 0], [5, 5, 5], [0, 8, 0]])
    X = np.vstack([c + np.random.randn(30, 3) * 0.8 for c in centers])
    y_true = np.repeat([1, 2, 3], 30)

    labels = hierarchical_cluster(X, n_clusters=3, method='ward')

    # 统计各簇样本数，并计算与真实标签的吻合度（列联表）
    print(f"样本总数：{len(X)}，设定簇数：3，链接方式：ward")
    for c in sorted(set(labels)):
        mask = (labels == c)
        # 该簇中占多数的真实类别
        majority = np.bincount(y_true[mask]).argmax()
        purity = np.mean(y_true[mask] == majority)
        print(f"  簇 {c}：样本数 {mask.sum()}，主要真实类别 {majority}，纯度 {purity:.4f}")

    # 画树状图
    plot_dendrogram(X, method='ward')
    print("已保存树状图：hierarchical_dendrogram.png")

    # 画聚类散点（取前两个特征）
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25)
    ax.set_xlabel('特征 1')
    ax.set_ylabel('特征 2')
    ax.set_title('层次聚类结果散点图（前两个特征）')
    plt.tight_layout()
    plt.savefig('hierarchical_demo.png', dpi=150)
    print("已保存聚类散点图：hierarchical_demo.png")
    plt.close(fig)
