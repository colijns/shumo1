# -*- coding: utf-8 -*-
"""
模块：basic_plots
功能：数学建模论文常用基础图模板——折线图（多系列+标注）、分组柱状图、堆叠柱状图、散点图（含回归线）、饼图
适用题型：通用（论文数据可视化必备，一个函数一张图，改数据即可套用）
依赖：numpy, matplotlib
用法：直接运行 `python basic_plots.py` 查看全部 demo；或 import 后调用绘图函数
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def plot_line(x, series_dict, title='多系列折线图', xlabel='x', ylabel='y',
              mark_extreme=True):
    """多系列折线图，带最值标注。

    参数
    ----
    x : array-like
        横坐标（各系列共用）。
    series_dict : dict
        {系列名: y 数组}，每个键对应一条折线。
    title, xlabel, ylabel : str
        图标题与坐标轴标签。
    mark_extreme : bool
        是否标注每条折线的最大值点。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', '^', 'D', 'v']  # 不同系列用不同 marker 便于黑白打印区分
    for idx, (name, y) in enumerate(series_dict.items()):
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, marker=markers[idx % len(markers)], ms=5, lw=1.8, label=name)
        if mark_extreme:
            # 标注该系列最大值点
            i_max = int(np.argmax(y))
            ax.annotate(f'{y[i_max]:.2f}', xy=(x[i_max], y[i_max]),
                        xytext=(0, 8), textcoords='offset points',
                        ha='center', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_grouped_bar(categories, series_dict, title='分组柱状图',
                     xlabel='', ylabel='数值'):
    """分组柱状图（多指标并列对比）。

    参数
    ----
    categories : list
        横轴类别标签（如年份、方案名）。
    series_dict : dict
        {系列名: 各类别数值数组}。
    title, xlabel, ylabel : str
        图标题与坐标轴标签。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    n_cat = len(categories)
    n_series = len(series_dict)
    x = np.arange(n_cat)
    width = 0.8 / n_series  # 每组总宽 0.8，均分给各系列

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, values) in enumerate(series_dict.items()):
        offset = (idx - (n_series - 1) / 2) * width  # 让各系列居中分布于类别两侧
        bars = ax.bar(x + offset, values, width * 0.9, label=name)
        ax.bar_label(bars, fmt='%.1f', fontsize=8)  # 柱顶标数值
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_stacked_bar(categories, series_dict, title='堆叠柱状图',
                     xlabel='', ylabel='数值'):
    """堆叠柱状图（构成占比随类别变化）。

    参数
    ----
    categories : list
        横轴类别标签。
    series_dict : dict
        {组成部分名: 各类别数值数组}，各组成部分逐层堆叠。
    title, xlabel, ylabel : str
        图标题与坐标轴标签。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(categories))  # 堆叠起点逐层累加
    for name, values in series_dict.items():
        values = np.asarray(values, dtype=float)
        ax.bar(x, values, 0.6, bottom=bottom, label=name)
        bottom += values
    # 标注各柱总量
    for xi, total in zip(x, bottom):
        ax.text(xi, total, f'{total:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scatter_regression(x, y, title='散点图与线性回归',
                            xlabel='自变量 x', ylabel='因变量 y'):
    """散点图 + 最小二乘回归线（含回归方程与 R² 标注）。

    参数
    ----
    x, y : array-like
        样本点横、纵坐标（长度一致）。
    title, xlabel, ylabel : str
        图标题与坐标轴标签。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # 一元线性回归（最小二乘）：y = k*x + b
    k, b = np.polyfit(x, y, 1)
    y_hat = k * x + b
    # 决定系数 R² = 1 - SS_res/SS_tot
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, c='steelblue', alpha=0.7, label='样本点')
    ax.plot(x, y_hat, 'r-', lw=2,
            label=f'回归线：y = {k:.4f}x + {b:.4f}（R² = {r2:.4f}）')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pie(labels, sizes, title='饼图（占比）', explode_max=True):
    """饼图（各部分占比），最大扇形可略微分离强调。

    参数
    ----
    labels : list
        各部分名称。
    sizes : array-like
        各部分数值（函数内自动换算百分比）。
    title : str
        图标题。
    explode_max : bool
        是否将最大扇形分离突出显示。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    sizes = np.asarray(sizes, dtype=float)
    explode = np.zeros(len(sizes))
    if explode_max:
        explode[int(np.argmax(sizes))] = 0.08  # 最大块略微分离

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(sizes, labels=labels, explode=explode, autopct='%.2f%%',
           startangle=90, counterclock=False,
           wedgeprops=dict(edgecolor='white', linewidth=1.5),
           textprops=dict(fontsize=10))
    ax.set_title(title)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    np.random.seed(42)  # 固定随机种子保证可复现
    print('=' * 55)
    print('基础图模板 demo：5 种常用图依次绘制')
    print('=' * 55)

    # ---------- 1) 折线图（多系列 + 最值标注） ----------
    print('\n【1】折线图：某市近三年月平均气温对比')
    months = np.arange(1, 13)
    temps = {
        '2022年': [2, 5, 10, 17, 23, 27, 30, 29, 24, 17, 9, 3],
        '2023年': [3, 6, 12, 18, 24, 28, 31, 30, 25, 18, 10, 4],
        '2024年': [1, 4, 11, 16, 22, 26, 29, 28, 23, 16, 8, 2],
    }
    fig1 = plot_line(months, temps, title='某市近三年月平均气温', xlabel='月份', ylabel='气温（℃）')
    fig1.savefig('basic_plots_line.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print('  已绘制 3 条折线（含最值标注），保存到 basic_plots_line.png。')

    # ---------- 2) 分组柱状图 ----------
    print('\n【2】分组柱状图：三个方案的四项指标得分')
    plans = ['方案一', '方案二', '方案三']
    scores = {
        '成本': [85, 78, 90],
        '质量': [92, 88, 80],
        '效率': [70, 85, 82],
        '环保': [88, 90, 75],
    }
    fig2 = plot_grouped_bar(plans, scores, title='三方案四项指标得分对比', ylabel='得分')
    fig2.savefig('basic_plots_grouped_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('  已绘制 3 类 × 4 系列分组柱状图，保存到 basic_plots_grouped_bar.png。')

    # ---------- 3) 堆叠柱状图 ----------
    print('\n【3】堆叠柱状图：某公司近三年营收构成')
    years = ['2022', '2023', '2024']
    revenue = {
        '产品A': [120, 150, 180],
        '产品B': [80, 95, 110],
        '产品C': [50, 70, 100],
    }
    fig3 = plot_stacked_bar(years, revenue, title='公司营收构成（万元）', ylabel='营收（万元）')
    fig3.savefig('basic_plots_stacked_bar.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print('  已绘制 3 层堆叠柱状图（柱顶为总量），保存到 basic_plots_stacked_bar.png。')

    # ---------- 4) 散点图 + 回归线 ----------
    print('\n【4】散点图 + 回归线：广告投入与销售额')
    ad_cost = np.linspace(10, 100, 30)                      # 广告投入（万元）
    sales = 3.5 * ad_cost + 50 + np.random.normal(0, 25, 30)  # 带噪声的线性关系
    fig4 = plot_scatter_regression(ad_cost, sales, title='广告投入与销售额关系',
                                   xlabel='广告投入（万元）', ylabel='销售额（万元）')
    fig4.savefig('basic_plots_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    k, b = np.polyfit(ad_cost, sales, 1)
    print(f'  拟合结果：斜率 = {k:.4f}，截距 = {b:.4f}（真实斜率 3.5，截距 50）。')

    # ---------- 5) 饼图 ----------
    print('\n【5】饼图：某电商市场份额')
    brands = ['品牌A', '品牌B', '品牌C', '品牌D', '其他']
    shares = [35, 25, 18, 12, 10]
    fig5 = plot_pie(brands, shares, title='某电商市场份额分布')
    fig5.savefig('basic_plots_pie.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print('  已绘制饼图（最大份额扇形分离强调），保存到 basic_plots_pie.png。')

    print('\n5 张基础图全部绘制完成。')
