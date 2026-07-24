# -*- coding: utf-8 -*-
"""
模块：correlation
功能：相关分析——Pearson/Spearman/Kendall 相关系数矩阵与 p 值矩阵、seaborn 热力图
适用题型：通用（变量相关性初探、指标筛选，国赛与华数杯统计分析部分常用）
依赖：numpy, pandas, scipy, matplotlib, seaborn
用法：直接运行 `python correlation.py` 查看 demo；或 import 后调用
      correlation_analysis / plot_corr_heatmap
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def correlation_analysis(df, method='pearson'):
    """计算数据框各列两两之间的相关系数与检验 p 值

    参数:
        df: pandas.DataFrame, 每一列为一个数值变量
        method: str, 相关系数类型：
            'pearson'  皮尔逊积矩相关（度量线性相关，要求近似正态）
            'spearman' 斯皮尔曼等级相关（基于秩次，度量单调关系，稳健）
            'kendall'  肯德尔等级相关（基于一致对，适合小样本）
    返回:
        dict:
            'coef'    相关系数矩阵（pandas.DataFrame）
            'p_value' 对应的显著性检验 p 值矩阵（pandas.DataFrame）
                      原假设 H0：两变量不相关，p < 0.05 认为相关显著
    """
    if method not in ('pearson', 'spearman', 'kendall'):
        raise ValueError("method 必须为 'pearson'、'spearman' 或 'kendall'")
    # 三种方法对应的 scipy 两样本相关检验函数
    func_map = {
        'pearson': stats.pearsonr,
        'spearman': stats.spearmanr,
        'kendall': stats.kendalltau,
    }
    corr_func = func_map[method]

    cols = df.columns
    n = len(cols)
    coef = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    p_value = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    # 逐对计算相关系数与 p 值（对称矩阵，只算上三角再镜像）
    for i in range(n):
        for j in range(i + 1, n):
            r, p = corr_func(df[cols[i]], df[cols[j]])
            coef.iloc[i, j] = coef.iloc[j, i] = r
            p_value.iloc[i, j] = p_value.iloc[j, i] = p
    return {'coef': coef, 'p_value': p_value}


def plot_corr_heatmap(coef, annot=True, cmap='RdBu_r'):
    """画相关系数矩阵热力图（seaborn）

    参数:
        coef: pandas.DataFrame, 相关系数矩阵（correlation_analysis 的返回项）
        annot: bool, 是否在格子里标注数值，默认 True
        cmap: str, 配色方案，默认红蓝对称色 'RdBu_r'
    """
    fig = plt.figure(figsize=(7, 6))
    sns.heatmap(coef, annot=annot, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, cbar_kws={'label': '相关系数'})
    plt.title('相关系数热力图')
    plt.tight_layout()
    fig.savefig('correlation_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # ---------------- demo：5 变量随机数据，三种方法对比 + 热力图 ----------------
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + np.random.normal(0, 0.5, n)   # 与 x1 强正相关
    x3 = -0.6 * x1 + np.random.normal(0, 0.8, n)  # 与 x1 中等负相关
    x4 = np.random.normal(0, 1, n)                # 与其他变量基本无关
    x5 = x2 ** 2 + np.random.normal(0, 0.5, n)    # 与 x2 非线性相关
    df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})

    print('=' * 60)
    print('相关分析 demo：5 变量，三种相关系数方法对比')
    print('=' * 60)

    result = None
    for method in ['pearson', 'spearman', 'kendall']:
        result = correlation_analysis(df, method=method)
        print(f'\n--- {method} 相关系数矩阵 ---')
        print(result['coef'].round(4))
        print(f'--- {method} p 值矩阵 ---')
        print(result['p_value'].round(4))

    # 用 pearson 结果画热力图
    result_pearson = correlation_analysis(df, method='pearson')
    plot_corr_heatmap(result_pearson['coef'])

    # 简单解读：找出 |r| > 0.5 且 p < 0.05 的变量对
    print('\n显著强相关变量对（|r| > 0.5 且 p < 0.05，pearson）：')
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = result_pearson['coef'].iloc[i, j]
            p = result_pearson['p_value'].iloc[i, j]
            if abs(r) > 0.5 and p < 0.05:
                print(f'  {cols[i]} 与 {cols[j]}：r = {r:.4f}，p = {p:.4f}')
    print('demo 运行结束。')
