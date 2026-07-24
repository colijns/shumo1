# -*- coding: utf-8 -*-
"""
模块：exponential_smoothing
功能：指数平滑时间序列预测（Holt-Winters，基于 statsmodels ExponentialSmoothing）
适用题型：通用高频（含趋势/季节性的时间序列预测，国赛C题、华数杯常用）
依赖：numpy, matplotlib, statsmodels
用法：直接运行 `python exponential_smoothing.py` 查看 demo；或 import 后调用 ets_forecast
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 抑制 statsmodels 的优化收敛警告
# 注意：必须在 import statsmodels 之后再设置，因为其导入时会重置 warnings 过滤器
warnings.filterwarnings('ignore')

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def ets_forecast(series, trend='add', seasonal=None, seasonal_periods=None, steps=5):
    """指数平滑（Holt-Winters）拟合并向前预测

    参数:
        series: array_like, 一维时间序列，长度建议 >= 2 个季节周期
        trend: str 或 None, 趋势成分类型：'add'（加法）、'mul'（乘法）、None（无趋势）
        seasonal: str 或 None, 季节成分类型：'add'、'mul'、None（无季节）
        seasonal_periods: int 或 None, 季节周期长度（seasonal 非 None 时必填），如月度数据取 12
        steps: int, 向前预测步数，默认 5
    返回:
        numpy.ndarray: 未来 steps 步的预测值
    """
    series = np.asarray(series, dtype=float).ravel()
    # 构造 Holt-Winters 指数平滑模型并极大似然估计平滑参数
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    fit = model.fit(optimized=True)
    forecast = np.asarray(fit.forecast(steps))
    return forecast


if __name__ == '__main__':
    # ---------------- demo：模拟季节序列（周期=4，如季度数据）预测 ----------------
    np.random.seed(42)
    n = 32  # 8 个完整季节周期
    t = np.arange(n)
    # 加法趋势 + 加法季节 + 小幅噪声
    seasonal_pattern = np.array([3.0, 5.0, 2.0, -1.0])  # 周期为 4 的季节模式
    series = 20 + 0.5 * t + np.tile(seasonal_pattern, n // 4) \
        + np.random.normal(0, 0.5, n)

    steps = 4
    print('=' * 60)
    print('指数平滑（Holt-Winters）预测 demo：季节周期 = 4')
    print('=' * 60)

    forecast = ets_forecast(series, trend='add', seasonal='add',
                            seasonal_periods=4, steps=steps)
    print(f'未来 {steps} 期预测值: {np.round(forecast, 4)}')

    # 画历史序列 + 拟合值 + 预测值（重新拟合一次以获得历史拟合值，仅用于画图）
    fit = ExponentialSmoothing(series, trend='add', seasonal='add',
                               seasonal_periods=4).fit()
    fitted = np.asarray(fit.fittedvalues)

    t_future = np.arange(n, n + steps)
    plt.figure(figsize=(9, 5))
    plt.plot(t, series, 'o-', markersize=4, label='历史序列')
    plt.plot(t, fitted, '--', color='gray', label='拟合值')
    plt.plot(np.concatenate([[n - 1], t_future]),
             np.concatenate([[series[-1]], forecast]),
             'r^--', label='预测值')
    plt.xlabel('时间 t')
    plt.ylabel('序列值')
    plt.title('Holt-Winters 指数平滑预测（加法趋势 + 加法季节，周期 4）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('exponential_smoothing_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('demo 运行结束。')
