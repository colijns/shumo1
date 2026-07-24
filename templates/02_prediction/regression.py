# -*- coding: utf-8 -*-
"""
模块：regression
功能：回归与曲线拟合——多元线性回归（sklearn/statsmodels）、多项式拟合、自定义曲线拟合
适用题型：通用高频（因素分析、趋势拟合、参数估计，国赛各题与华数杯常用）
依赖：numpy, scipy, matplotlib, sklearn, statsmodels
用法：直接运行 `python regression.py` 查看 demo；或 import 后调用
      linear_regression / poly_fit / curve_fit_custom
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def linear_regression(X, y, show_ols_summary=True):
    """多元线性回归：sklearn 拟合并打印回归系数与 R²，可选打印 statsmodels OLS 摘要

    参数:
        X: array_like, 自变量矩阵，shape (n_samples, n_features)
        y: array_like, 因变量向量，shape (n_samples,)
        show_ols_summary: bool, 是否打印 statsmodels OLS 统计摘要（含 t/F 检验），默认 True
    返回:
        dict:
            'coef'      回归系数（numpy 数组）
            'intercept' 截距
            'r2'        决定系数 R²
            'model'     拟合好的 sklearn LinearRegression 对象（可用于 predict）
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    print('回归系数 coef:', np.round(model.coef_, 4))
    print(f'截距 intercept: {model.intercept_:.4f}')
    print(f'决定系数 R^2: {r2:.4f}')

    if show_ols_summary:
        # statsmodels OLS 提供显著性检验等完整统计信息
        import statsmodels.api as sm
        X_with_const = sm.add_constant(X)  # 添加常数项列
        ols_res = sm.OLS(y, X_with_const).fit()
        print(ols_res.summary())

    return {
        'coef': model.coef_,
        'intercept': float(model.intercept_),
        'r2': float(r2),
        'model': model,
    }


def poly_fit(x, y, deg=2, plot=True):
    """一维多项式拟合（np.polyfit）并画图

    参数:
        x: array_like, 自变量
        y: array_like, 因变量
        deg: int, 多项式次数，默认 2
        plot: bool, 是否画拟合对比图，默认 True
    返回:
        numpy.ndarray: 多项式系数（降幂排列，可用 np.polyval 求值）
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    coeffs = np.polyfit(x, y, deg)
    y_fit = np.polyval(coeffs, x)
    # 计算拟合优度 R² = 1 - SSE/SST
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f'{deg} 次多项式系数（降幂）:', np.round(coeffs, 4))
    print(f'多项式拟合 R^2: {r2:.4f}')

    if plot:
        # 用密集采样点画光滑的拟合曲线
        x_dense = np.linspace(x.min(), x.max(), 200)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='steelblue', label='原始数据')
        plt.plot(x_dense, np.polyval(coeffs, x_dense), 'r-',
                 label=f'{deg} 次多项式拟合')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'多项式拟合（deg={deg}，R^2={r2:.4f}）')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig('regression_poly_demo.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    return coeffs


def curve_fit_custom(x, y, func, p0):
    """自定义函数曲线拟合（scipy.optimize.curve_fit 非线性最小二乘）

    参数:
        x: array_like, 自变量
        y: array_like, 因变量
        func: callable, 形如 func(x, *params) 的自定义模型函数
        p0: sequence, 参数初值猜测
    返回:
        dict:
            'params' 最优参数估计值（numpy 数组）
            'pcov'   参数协方差矩阵（对角线开方即参数标准误）
            'y_fit'  拟合值
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    popt, pcov = curve_fit(func, x, y, p0=p0, maxfev=10000)
    y_fit = func(x, *popt)
    perr = np.sqrt(np.diag(pcov))  # 参数标准误
    print('拟合参数:', np.round(popt, 4))
    print('参数标准误:', np.round(perr, 4))
    return {'params': popt, 'pcov': pcov, 'y_fit': y_fit}


if __name__ == '__main__':
    np.random.seed(42)

    # ---------------- 案例 1：多元线性回归 ----------------
    print('=' * 60)
    print('案例 1：多元线性回归（真实模型 y = 3 + 2*x1 - 1.5*x2 + 噪声）')
    print('=' * 60)
    n = 50
    X = np.column_stack([np.random.uniform(0, 10, n),
                         np.random.uniform(0, 10, n)])
    y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.normal(0, 0.5, n)
    linear_regression(X, y)

    # ---------------- 案例 2：多项式拟合 ----------------
    print('\n' + '=' * 60)
    print('案例 2：多项式拟合（真实模型 y = 0.5x^2 + x + 2 + 噪声）')
    print('=' * 60)
    x2 = np.linspace(0, 10, 40)
    y2 = 0.5 * x2 ** 2 + x2 + 2 + np.random.normal(0, 3, len(x2))
    poly_fit(x2, y2, deg=2)

    # ---------------- 案例 3：自定义指数曲线拟合 ----------------
    print('\n' + '=' * 60)
    print('案例 3：自定义曲线拟合（真实模型 y = 2*exp(0.3x) + 1 + 噪声）')
    print('=' * 60)

    def exp_func(x, a, b, c):
        """指数增长模型 y = a * exp(b * x) + c"""
        return a * np.exp(b * x) + c

    x3 = np.linspace(0, 5, 40)
    y3 = 2 * np.exp(0.3 * x3) + 1 + np.random.normal(0, 0.3, len(x3))
    result3 = curve_fit_custom(x3, y3, exp_func, p0=[1, 0.2, 0])
    a_hat, b_hat, c_hat = result3['params']
    print(f'参数估计：a = {a_hat:.4f}，b = {b_hat:.4f}，c = {c_hat:.4f}（真值 2, 0.3, 1）')

    # 画自定义拟合对比图
    x3_dense = np.linspace(0, 5, 200)
    plt.figure(figsize=(8, 5))
    plt.scatter(x3, y3, color='steelblue', label='原始数据')
    plt.plot(x3_dense, exp_func(x3_dense, *result3['params']), 'r-',
             label=f'指数拟合 y={a_hat:.2f}e^({b_hat:.2f}x)+{c_hat:.2f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('自定义指数曲线拟合')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('regression_curve_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('demo 运行结束。')
