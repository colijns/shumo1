"""
模块：pca_evaluation
功能：PCA 主成分降维与综合评价——按累计方差贡献率选主成分，加权合成综合得分并排名
适用题型：国赛C题高频 / 华数杯高频（多指标降维、客观综合评价）
依赖：numpy, scikit-learn, matplotlib（仅 demo 绘图用）
用法：直接运行 `python pca_evaluation.py` 查看 demo；或 import 后调用 pca_evaluate(X, threshold)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def pca_evaluate(X, threshold=0.85):
    """
    PCA 综合评价：标准化 -> 主成分提取 -> 按贡献率加权合成综合得分。

    计算步骤：
        1. Z-score 标准化消除量纲；
        2. 对标准化数据做 PCA，按累计方差贡献率 >= threshold 选取前 k 个主成分；
        3. 综合得分 F = Σ_j (ratio_j / Σ ratio) * F_j，即前 k 个主成分得分的加权平均
           （权重为归一化的方差贡献率）；
        4. 载荷矩阵 loading_ij = 第 i 个原始指标与第 j 个主成分的相关系数。

    参数：
        X         : numpy.ndarray 或 pandas.DataFrame，(n 样本 × m 指标) 原始数据
        threshold : float，累计方差贡献率阈值，默认 0.85（即 85%）
    返回：
        dict：
            scores          : (n,) numpy.ndarray，各样本综合得分（越大越优）
            ranking         : (n,) numpy.ndarray，各样本名次（1 为最优）
            n_components    : int，选取的主成分个数 k
            explained_ratio : (k,) numpy.ndarray，前 k 个主成分的方差贡献率
            loadings        : (m, k) numpy.ndarray，主成分载荷矩阵
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X 必须是 (n 样本 × m 指标) 的二维矩阵')
    n, m = X.shape

    # 1. Z-score 标准化
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)     # 防止除零
    Z = (X - mu) / sigma

    # 2. 全成分 PCA
    pca = PCA()
    F = pca.fit_transform(Z)                     # (n, m) 各主成分得分
    ratio = pca.explained_variance_ratio_        # 各主成分方差贡献率
    cum = np.cumsum(ratio)                       # 累计方差贡献率

    # 3. 按阈值选主成分个数
    k = int(np.searchsorted(cum, threshold) + 1)
    k = min(k, m)

    # 4. 综合得分：前 k 个主成分按归一化贡献率加权平均
    w_ratio = ratio[:k] / cum[k - 1]
    scores = (F[:, :k] * w_ratio).sum(axis=1)

    # 5. 载荷矩阵：特征向量 × sqrt(特征值)，即原始指标与主成分的相关系数
    loadings = pca.components_.T[:, :k] * np.sqrt(pca.explained_variance_[:k])

    # 6. 排名：得分越高名次越靠前
    ranking = scores.argsort()[::-1].argsort() + 1

    return {'scores': scores,
              'ranking': ranking,
              'n_components': k,
              'explained_ratio': ratio[:k],
              'loadings': loadings}


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：10 个样本 × 8 个指标的综合评价 =====
    # 构造含 3 个潜在因子的相关指标数据（模拟经济发展水平评价）
    n, m = 10, 8
    latent = np.random.randn(n, 3)                       # 3 个潜在因子
    coef = np.random.randn(3, m)
    X = latent @ coef + np.random.randn(n, m) * 0.3      # 观测指标 = 因子线性组合 + 噪声
    X = np.abs(X) * 10 + 50                              # 平移为正数，模拟实际指标

    indicator_names = [f'指标{i + 1}' for i in range(m)]
    sample_names = [f'样本{i + 1}' for i in range(n)]

    print('===== PCA 综合评价 demo（10 样本 × 8 指标）=====')
    print('原始数据（前 3 行）：')
    print(np.round(X[:3], 4))

    res = pca_evaluate(X, threshold=0.85)

    print(f"\n选取主成分个数 k = {res['n_components']}（累计方差贡献率 ≥ 85%）")
    print('各主成分方差贡献率：')
    for j, r in enumerate(res['explained_ratio']):
        print(f'  主成分{j + 1}：{r:.4f}')
    print(f"累计贡献率 = {res['explained_ratio'].sum():.4f}")

    print('\n主成分载荷矩阵（行=指标，列=主成分）：')
    print(np.round(res['loadings'], 4))

    print('\n----- 各样本综合得分与排名 -----')
    for i in range(n):
        print(f"{sample_names[i]}：得分 = {res['scores'][i]: .4f}，名次 = {res['ranking'][i]}")
    best = int(np.argmax(res['scores']))
    print(f"\n结论：综合得分最高的是 {sample_names[best]}"
          f"（得分 {res['scores'][best]:.4f}）")

    # ---- 可视化 1：方差贡献率碎石图（展示全部 m 个主成分）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xk = np.arange(1, m + 1)
    # 用标准化数据重跑一次全成分 PCA 取完整贡献率
    from sklearn.decomposition import PCA as _PCA
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))
    ratio_full = _PCA().fit(Z).explained_variance_ratio_
    axes[0].bar(xk, ratio_full, color='#4C72B0', edgecolor='black', label='单个贡献率')
    axes[0].plot(xk, np.cumsum(ratio_full), 'o-', color='#DD8452', label='累计贡献率')
    axes[0].axhline(0.85, color='gray', linestyle='--', linewidth=1, label='85% 阈值')
    axes[0].set_title('PCA 碎石图')
    axes[0].set_xlabel('主成分编号')
    axes[0].set_ylabel('方差贡献率')
    axes[0].legend()

    # ---- 可视化 2：综合得分柱状图 ----
    axes[1].bar(sample_names, res['scores'], color='#4C72B0', edgecolor='black')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('各样本 PCA 综合得分')
    axes[1].set_ylabel('综合得分')
    plt.setp(axes[1].get_xticklabels(), rotation=45)
    plt.tight_layout()
    fig.savefig('pca_evaluation_demo.png', dpi=150, bbox_inches='tight')
    print('pca_evaluation_demo.png 已保存')
    plt.close(fig)
    print('\npca_evaluation demo 运行完毕')
