# -*- coding: utf-8 -*-
"""
模块：descriptive
功能：描述性统计分析——均值/中位数/标准差/方差/偏度/峰度/极值汇总表与直方图
适用题型：通用（数据探索性分析 EDA 第一步，各类赛题通用）
依赖：numpy, pandas, scipy, matplotlib
用法：直接运行 `python descriptive.py` 查看 demo；或 import 后调用 describe_all
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def describe_all(df):
    """计算数据框每一列的常用描述性统计量

    参数:
        df: pandas.DataFrame, 每一列为一个数值变量
    返回:
        pandas.DataFrame: 行索引为统计量（均值/中位数/标准差/方差/偏度/峰度/
                          最小值/最大值/极差），列索引与原 df 相同
    """
    result = pd.DataFrame({
        '均值': df.mean(),
        '中位数': df.median(),
        '标准差': df.std(),
        '方差': df.var(),
        '偏度': stats.skew(df, axis=0, bias=False),      # 偏度：衡量分布不对称性
        '峰度': stats.kurtosis(df, axis=0, bias=False),  # 超额峰度：正态分布为 0
        '最小值': df.min(),
        '最大值': df.max(),
        '极差': df.max() - df.min(),
    }).T  # 转置：统计量作行、变量作列，便于阅读
    return result


def plot_histograms(df, bins=15):
    """为数据框每一列画直方图（含正态密度参考曲线）

    参数:
        df: pandas.DataFrame, 每一列为一个数值变量
        bins: int, 直方图分箱数，默认 15
    """
    cols = df.columns
    n_cols = len(cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = df[col].dropna().values
        ax.hist(data, bins=bins, density=True, color='steelblue',
                alpha=0.7, edgecolor='white', label='频率直方图')
        # 叠加用样本均值/标准差构造的正态密度曲线作参照
        x_dense = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_dense, stats.norm.pdf(x_dense, data.mean(), data.std()),
                'r-', label='正态密度参考')
        ax.set_title(f'{col} 分布直方图')
        ax.set_xlabel(col)
        ax.set_ylabel('密度')
        ax.legend()
    plt.tight_layout()
    fig.savefig('descriptive_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # ---------------- demo：随机数据出描述性统计表 + 直方图 ----------------
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        '成绩': np.random.normal(75, 10, n),          # 近似正态
        '身高cm': np.random.normal(170, 6, n),        # 近似正态
        '收入千元': np.random.exponential(8, n) + 3,  # 右偏分布
    })

    print('=' * 60)
    print('描述性统计 demo')
    print('=' * 60)
    table = describe_all(df)
    print(table.round(4))
    print('\n说明：偏度 > 0 为右偏，< 0 为左偏；峰度为超额峰度，正态分布约为 0。')

    # 画各变量直方图
    plot_histograms(df)
    print('demo 运行结束。')
