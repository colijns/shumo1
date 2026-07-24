# -*- coding: utf-8 -*-
"""
模块：interpolation_fitting
功能：插值与拟合——interp1d 一维插值（linear/cubic）、griddata 二维散乱点网格化、np.polyfit 多项式拟合
适用题型：通用高频（数据补全、曲面重构、经验公式拟合，国赛A/B题与华数杯常用）
依赖：numpy, scipy, matplotlib
用法：直接运行 `python interpolation_fitting.py` 查看 demo；或 import 后调用三个案例函数
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def demo_interp1d():
    """案例 1：一维插值——对稀疏采样点分别做线性插值与三次插值并对比

    返回:
        tuple: (f_linear, f_cubic) 两个 interp1d 插值函数，可对任意 x 求值
    """
    print('=' * 60)
    print('案例 1：interp1d 一维插值（linear vs cubic）')
    print('=' * 60)
    # 稀疏采样点（真实曲线为 sin + 小幅扰动）
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = np.sin(x) + 0.1 * x

    # 构造线性插值与三次插值函数（fill_value='extrapolate' 允许外推）
    f_linear = interp1d(x, y, kind='linear', fill_value='extrapolate')
    f_cubic = interp1d(x, y, kind='cubic', fill_value='extrapolate')

    # 在密集网格上求值并画图
    x_dense = np.linspace(0, 10, 200)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'ko', markersize=8, label='原始采样点')
    plt.plot(x_dense, f_linear(x_dense), '--', label='线性插值 linear')
    plt.plot(x_dense, f_cubic(x_dense), '-', label='三次插值 cubic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('一维插值对比：linear vs cubic')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('interp1d_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 演示：用插值函数估计中间点的值
    x_new = np.array([1.5, 4.5, 7.5])
    print('估计点 x =', x_new)
    print('线性插值结果:', np.round(f_linear(x_new), 4))
    print('三次插值结果:', np.round(f_cubic(x_new), 4))
    return f_linear, f_cubic


def demo_griddata():
    """案例 2：griddata 二维散乱点网格化插值并画等高线图

    返回:
        tuple: (grid_x, grid_y, grid_z) 网格化后的坐标矩阵与插值结果
    """
    print('=' * 60)
    print('案例 2：griddata 二维散乱点网格化插值')
    print('=' * 60)
    np.random.seed(42)
    # 随机散乱采样点，真实曲面 z = sin(x) * cos(y)
    n_points = 200
    x = np.random.uniform(-3, 3, n_points)
    y = np.random.uniform(-3, 3, n_points)
    z = np.sin(x) * np.cos(y) + np.random.normal(0, 0.05, n_points)

    # 构造规则网格，把散乱点插值到网格上（cubic 方法）
    grid_x, grid_y = np.mgrid[-3:3:100j, -3:3:100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # 画等高线填充图 + 原始散乱点
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(grid_x, grid_y, grid_z, levels=15, cmap='viridis')
    plt.colorbar(cs, label='z 值')
    plt.scatter(x, y, c='red', s=5, alpha=0.5, label='散乱采样点')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('griddata 二维散乱点网格化（cubic）')
    plt.legend()
    plt.tight_layout()
    plt.savefig('griddata_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 演示：估计指定点的函数值（cubic 插值；查询点落在训练范围内）
    pts_new = np.array([[0.5, 0.5], [-1.0, 1.0]])
    z_new = griddata((x, y), z, pts_new, method='cubic')
    print('估计点:', pts_new.tolist())
    print('griddata 插值结果:', np.round(z_new, 4))
    print('真实值 sin(x)cos(y):', np.round(np.sin(pts_new[:, 0]) * np.cos(pts_new[:, 1]), 4))
    return grid_x, grid_y, grid_z


def demo_polyfit(deg=3):
    """案例 3：np.polyfit 多项式拟合（对带噪声数据拟合经验公式）

    参数:
        deg: int, 多项式次数，默认 3
    返回:
        numpy.ndarray: 多项式系数（降幂排列）
    """
    print('=' * 60)
    print(f'案例 3：np.polyfit 多项式拟合（deg = {deg}）')
    print('=' * 60)
    np.random.seed(42)
    # 真实关系 y = 0.1x^3 - 0.5x^2 + x + 5 + 噪声
    x = np.linspace(-2, 6, 50)
    y = 0.1 * x ** 3 - 0.5 * x ** 2 + x + 5 + np.random.normal(0, 0.8, len(x))

    coeffs = np.polyfit(x, y, deg)
    y_fit = np.polyval(coeffs, x)
    # 拟合优度 R^2
    r2 = 1 - np.sum((y - y_fit) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f'{deg} 次多项式系数（降幂）:', np.round(coeffs, 4))
    print(f'拟合优度 R^2 = {r2:.4f}（真实系数为 0.1, -0.5, 1, 5）')

    x_dense = np.linspace(-2, 6, 200)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='steelblue', label='带噪声数据')
    plt.plot(x_dense, np.polyval(coeffs, x_dense), 'r-',
             label=f'{deg} 次多项式拟合')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'多项式拟合（R^2 = {r2:.4f}）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('polyfit_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    return coeffs


if __name__ == '__main__':
    demo_interp1d()
    demo_griddata()
    demo_polyfit(deg=3)
    print('demo 运行结束。')
