# -*- coding: utf-8 -*-
"""
模块：statistical_plots
功能：统计图模板——相关系数热力图（seaborn）、箱线图、小提琴图、雷达图（matplotlib 极坐标手写）
适用题型：通用（数据探索性分析 EDA、多方案多维指标对比展示）
依赖：numpy, pandas, matplotlib, seaborn
用法：直接运行 `python statistical_plots.py` 查看全部 demo；或 import 后调用绘图函数
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def plot_heatmap(corr, title='相关系数热力图', annot=True, cmap='RdBu_r'):
    """相关系数（或任意方阵）热力图。

    参数
    ----
    corr : pandas.DataFrame 或 2D array-like
        相关系数矩阵（方阵）。DataFrame 的行列名将作为刻度标签。
    title : str
        图标题。
    annot : bool
        是否在格子内标注数值。
    cmap : str
        配色方案（默认 RdBu_r：正相关红、负相关蓝）。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=annot, fmt='.2f', cmap=cmap, center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_boxplot(df, title='箱线图', ylabel='数值'):
    """箱线图：展示各列数据的分布、离群点与中位数。

    参数
    ----
    df : pandas.DataFrame
        每列为一组数据，列名为横轴标签。
    title, ylabel : str
        图标题与纵轴标签。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # '组名': 长格式列名，便于 seaborn 分组
    sns.boxplot(data=df, ax=ax, palette='Set2', width=0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_violin(df, title='小提琴图', ylabel='数值'):
    """小提琴图：箱线图 + 核密度估计，展示分布形状细节。

    参数
    ----
    df : pandas.DataFrame
        每列为一组数据，列名为横轴标签。
    title, ylabel : str
        图标题与纵轴标签。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # inner='box' 在小提琴内部叠加迷你箱线图
    sns.violinplot(data=df, ax=ax, palette='Set3', inner='box')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_radar(labels, series_dict, title='雷达图（多维指标对比）'):
    """雷达图（蜘蛛图）：matplotlib 极坐标手写实现，用于多方案多维指标对比。

    参数
    ----
    labels : list
        各维度指标名称（建议 3~8 维）。
    series_dict : dict
        {方案名: 各维度数值数组}，各数组长度须与 labels 一致。
    title : str
        图标题。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    n_dim = len(labels)
    # 每个维度的极角：圆周等分；闭合曲线需把起点拼到末尾
    angles = np.linspace(0, 2 * np.pi, n_dim, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    for name, values in series_dict.items():
        values = list(values) + list(values)[:1]  # 首尾相接形成闭合多边形
        ax.plot(angles, values, 'o-', lw=2, ms=4, label=name)
        ax.fill(angles, values, alpha=0.15)

    # 维度标签放到对应极角位置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(title, pad=20)  # pad 避免标题与顶部标签重叠
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    np.random.seed(42)  # 固定随机种子保证可复现
    print('=' * 55)
    print('统计图模板 demo：热力图/箱线图/小提琴图/雷达图')
    print('=' * 55)

    # ---------- 1) 相关系数热力图 ----------
    print('\n【1】相关系数热力图：5 个经济指标')
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n)   # 与 x1 强正相关
    x3 = -0.5 * x1 + 0.8 * np.random.normal(0, 1, n)  # 与 x1 中等负相关
    x4 = np.random.normal(0, 1, n)                     # 与前面基本无关
    x5 = 0.3 * x2 + 0.9 * np.random.normal(0, 1, n)   # 与 x2 弱正相关
    df_corr = pd.DataFrame({'GDP增速': x1, '消费指数': x2, '失业率': x3,
                            '汇率波动': x4, '居民收入': x5})
    corr = df_corr.corr()
    fig1 = plot_heatmap(corr, title='5 个经济指标相关系数热力图')
    fig1.savefig('statistical_plots_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('  相关系数矩阵：')
    print(corr.to_string(float_format=lambda v: f'{v:.4f}'))

    # ---------- 2) 箱线图 ----------
    print('\n【2】箱线图：三个车间产品尺寸分布')
    df_box = pd.DataFrame({
        '车间甲': np.random.normal(10.0, 0.3, 100),
        '车间乙': np.random.normal(10.2, 0.5, 100),
        '车间丙': np.concatenate([np.random.normal(9.8, 0.2, 95),
                                  np.random.normal(11.0, 0.1, 5)]),  # 混入离群点
    })
    fig2 = plot_boxplot(df_box, title='三个车间产品尺寸箱线图', ylabel='尺寸（mm）')
    fig2.savefig('statistical_plots_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'  各车间均值：{df_box.mean().round(4).to_dict()}')
    print(f'  各车间标准差：{df_box.std().round(4).to_dict()}')

    # ---------- 3) 小提琴图 ----------
    print('\n【3】小提琴图：三个班级成绩分布形状')
    df_violin = pd.DataFrame({
        '一班': np.random.normal(80, 8, 120),
        '二班': np.concatenate([np.random.normal(70, 5, 60),
                                np.random.normal(90, 5, 60)]),  # 双峰分布
        '三班': np.random.normal(75, 12, 120),
    })
    fig3 = plot_violin(df_violin, title='三个班级成绩分布小提琴图', ylabel='成绩（分）')
    fig3.savefig('statistical_plots_violin.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print('  二班为双峰分布（小提琴图比箱线图更能显示分布形状）。')

    # ---------- 4) 雷达图 ----------
    print('\n【4】雷达图：三个候选人六项能力评估')
    dims = ['专业能力', '沟通协作', '创新思维', '执行效率', '抗压能力', '学习能力']
    candidates = {
        '候选人甲': [90, 75, 85, 80, 70, 88],
        '候选人乙': [78, 90, 72, 85, 82, 80],
        '候选人丙': [85, 82, 90, 76, 88, 84],
    }
    fig4 = plot_radar(dims, candidates, title='候选人六项能力雷达图')
    fig4.savefig('statistical_plots_radar.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    for name, vals in candidates.items():
        print(f'  {name} 综合均分：{np.mean(vals):.4f}')

    print('\n4 张统计图全部绘制完成。')
