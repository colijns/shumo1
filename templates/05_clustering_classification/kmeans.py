# -*- coding: utf-8 -*-
"""
模块：kmeans
功能：K-means 聚类，结合肘部法（SSE）与轮廓系数自动选择最佳簇数 k
适用题型：通用高频（客户分群、样本分类预处理、数据探索）
依赖：numpy, scikit-learn, matplotlib
用法：直接运行 `python kmeans.py` 查看 demo；或 import 后调用 kmeans_auto
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def kmeans_auto(X, k_range=range(2, 8), seed=None):
    """
    对数据 X 在 k_range 范围内逐一做 K-means，用轮廓系数最大原则自动选 best_k。

    参数：
        X       : 样本数据，shape (n_samples, n_features)
        k_range : 候选簇数范围（可迭代整数，如 range(2, 8)）
        seed    : 随机种子，传入后固定 KMeans 初始化与 numpy 随机状态

    返回：
        dict，键为：
            'best_k'         : 最佳簇数（轮廓系数最大者）
            'labels'         : best_k 对应的聚类标签，shape (n_samples,)
            'centers'        : best_k 对应的簇中心，shape (best_k, n_features)
            'sse_list'       : 各 k 的簇内平方和（肘部法依据），与 k_range 等长
            'silhouette_list': 各 k 的平均轮廓系数，与 k_range 等长
            'k_list'         : 实际使用的 k 值列表
    """
    X = np.asarray(X, dtype=float)
    if seed is not None:
        np.random.seed(seed)
    k_list = list(k_range)
    sse_list, silhouette_list = [], []
    models = {}

    for k in k_list:
        # n_init=10 多次初始化取最优，random_state 固定保证可复现
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        sse_list.append(float(km.inertia_))                    # 簇内平方和（肘部法）
        silhouette_list.append(float(silhouette_score(X, labels)))
        models[k] = km

    # 轮廓系数最大者即为最佳 k
    best_k = k_list[int(np.argmax(silhouette_list))]
    best_model = models[best_k]
    return {
        'best_k': best_k,
        'labels': best_model.labels_,
        'centers': best_model.cluster_centers_,
        'sse_list': sse_list,
        'silhouette_list': silhouette_list,
        'k_list': k_list,
    }


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现

    from sklearn.datasets import make_blobs

    print("=" * 60)
    print("K-means demo：make_blobs 合成数据自动选 k")
    print("=" * 60)
    # 真实簇数为 4 的二维合成数据
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8,
                           random_state=42)

    res = kmeans_auto(X, k_range=range(2, 8), seed=42)

    print("各 k 值评估结果：")
    for k, sse, sil in zip(res['k_list'], res['sse_list'], res['silhouette_list']):
        print(f"  k={k}：SSE={sse:.4f}，平均轮廓系数={sil:.4f}")
    print(f"自动选择的最佳簇数：best_k = {res['best_k']}（真实簇数为 4）")
    print(f"簇中心坐标：\n{np.round(res['centers'], 4)}")

    # 可视化：肘部法 + 轮廓系数 + 聚类散点
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(res['k_list'], res['sse_list'], 'bo-')
    axes[0].set_xlabel('簇数 k')
    axes[0].set_ylabel('簇内平方和 SSE')
    axes[0].set_title('肘部法选 k（找拐点）')
    axes[0].grid(alpha=0.3)

    axes[1].plot(res['k_list'], res['silhouette_list'], 'gs-')
    axes[1].axvline(res['best_k'], color='r', linestyle='--',
                    label=f'best_k={res["best_k"]}')
    axes[1].set_xlabel('簇数 k')
    axes[1].set_ylabel('平均轮廓系数')
    axes[1].set_title('轮廓系数选 k（越大越好）')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    sc = axes[2].scatter(X[:, 0], X[:, 1], c=res['labels'], cmap='viridis', s=15)
    axes[2].scatter(res['centers'][:, 0], res['centers'][:, 1],
                    c='red', marker='X', s=200, edgecolors='k', label='簇中心')
    axes[2].set_xlabel('特征 1')
    axes[2].set_ylabel('特征 2')
    axes[2].set_title(f'K-means 聚类结果（k={res["best_k"]}）')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('kmeans_demo.png', dpi=150)
    print("已保存肘部图/轮廓系数图/聚类散点图：kmeans_demo.png")
    plt.close(fig)
