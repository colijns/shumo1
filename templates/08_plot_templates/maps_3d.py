# -*- coding: utf-8 -*-
"""
模块：maps_3d
功能：3D 曲面图、等高线（填充）图（matplotlib）；pyecharts 中国地图（可选依赖，未安装时优雅跳过，不影响其余部分运行）
适用题型：通用（二元函数/响应曲面可视化、地形与空间分布展示、分省数据地图展示）
依赖：numpy, matplotlib；可选：pyecharts（未安装则地图部分自动跳过）
用法：直接运行 `python maps_3d.py` 查看 demo；或 import 后调用绘图函数
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  注册 3D 投影，勿删

# 中文字体与负号正常显示（按 SPEC 约定两行 rcParams）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def plot_surface(X, Y, Z, title='3D 曲面图', xlabel='X', ylabel='Y', zlabel='Z',
                 cmap='viridis'):
    """3D 曲面图（二元函数 z = f(x, y) 可视化）。

    参数
    ----
    X, Y : numpy.ndarray
        np.meshgrid 生成的网格坐标矩阵。
    Z : numpy.ndarray
        网格点处的函数值（形状与 X、Y 一致）。
    title, xlabel, ylabel, zlabel : str
        图标题与各坐标轴标签。
    cmap : str
        曲面配色方案。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    # rstride/cstride 控制网格抽样步长，数值小则更细腻
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label=zlabel)
    fig.tight_layout()
    return fig


def plot_contour(X, Y, Z, title='等高线（填充）图', xlabel='X', ylabel='Y',
                 levels=15, cmap='viridis'):
    """等高线 + 填充图（二元函数的平面投影表示）。

    参数
    ----
    X, Y : numpy.ndarray
        np.meshgrid 生成的网格坐标矩阵。
    Z : numpy.ndarray
        网格点处的函数值。
    title, xlabel, ylabel : str
        图标题与坐标轴标签。
    levels : int
        等高线分级数。
    cmap : str
        填充配色方案。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    # 先填充再画等高线线条，层次更清楚
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    cl = ax.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.4, alpha=0.6)
    ax.clabel(cl, inline=True, fontsize=8, fmt='%.2f')  # 等高线数值标注
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(cf, ax=ax, label='函数值 Z')
    fig.tight_layout()
    return fig


def plot_china_map_pyecharts(province_data, title='分省数据中国地图'):
    """pyecharts 中国分省地图（交互式 HTML，可选依赖）。

    参数
    ----
    province_data : list of tuple
        [(省份名, 数值), ...]，省份名须为 pyecharts 认可的标准名（如 '广东'）。
    title : str
        地图标题。

    返回
    ----
    ok : bool
        True 表示已生成 HTML 文件；False 表示未安装 pyecharts 已跳过。
    """
    # 可选依赖保护：未安装 pyecharts 时打印提示并优雅跳过（不抛异常）
    try:
        from pyecharts.charts import Map
        from pyecharts import options as opts
    except ImportError:
        print('未安装 pyecharts，可 pip install pyecharts，本 demo 跳过中国地图部分。')
        return False

    # 构造地图并绑定数据
    m = Map()
    m.add('指标值', province_data, 'china')
    m.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        visualmap_opts=opts.VisualMapOpts(is_piecewise=False),  # 连续色阶
    )
    out_path = 'china_map.html'
    m.render(out_path)
    print(f'中国地图已生成：{out_path}（用浏览器打开查看交互效果）')
    return True


if __name__ == '__main__':
    np.random.seed(42)  # 固定随机种子保证可复现
    print('=' * 55)
    print('3D 图与地图 demo：3D 曲面 / 等高线 / 中国地图')
    print('=' * 55)

    # ---------- 1) 3D 曲面图 ----------
    # 示例函数：双峰曲面 z = 峰1 - 峰2 + 波动（模拟地形/响应曲面）
    print('\n【1】3D 曲面图：二元函数响应曲面')
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (3 * np.exp(-(X ** 2 + Y ** 2))
         - 2 * np.exp(-((X - 1.5) ** 2 + (Y - 1.5) ** 2))
         + 0.3 * np.sin(2 * X) * np.cos(2 * Y))
    fig1 = plot_surface(X, Y, Z, title='二元函数 3D 响应曲面', zlabel='Z 值')
    fig1.savefig('maps_3d_surface.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f'  曲面取值范围：[{Z.min():.4f}, {Z.max():.4f}]，'
          f'峰值位于 ({x[np.unravel_index(np.argmax(Z), Z.shape)[1]]:.4f}, '
          f'{y[np.unravel_index(np.argmax(Z), Z.shape)[0]]:.4f})')

    # ---------- 2) 等高线（填充）图 ----------
    print('\n【2】等高线图：同一二元函数的平面投影')
    fig2 = plot_contour(X, Y, Z, title='二元函数等高线（填充）图')
    fig2.savefig('maps_3d_contour.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print('  已绘制填充等高线并标注数值，保存到 maps_3d_contour.png。')

    # ---------- 3) pyecharts 中国地图（可选依赖保护） ----------
    print('\n【3】pyecharts 中国地图：分省 GDP 示意数据')
    province_gdp = [
        ('广东', 12.9), ('江苏', 12.3), ('山东', 9.2), ('浙江', 8.3),
        ('河南', 6.1), ('四川', 6.0), ('湖北', 5.6), ('福建', 5.4),
        ('湖南', 5.1), ('北京', 4.4), ('上海', 4.7), ('陕西', 3.4),
    ]
    ok = plot_china_map_pyecharts(province_gdp, title='分省 GDP 示意地图（万亿元）')
    if not ok:
        print('  （matplotlib 的 3D 曲面与等高线图不受影响，均已正常绘制）')

    print('\ndemo 运行结束。')
