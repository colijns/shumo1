# -*- coding: utf-8 -*-
"""
模块：arima_model
功能：ARIMA(p,d,q) 时间序列预测，AIC 网格搜索自动定阶 + 预测（含置信区间）
适用题型：通用高频（平稳/差分平稳时间序列预测，国赛C题、华数杯常用）
依赖：numpy, matplotlib, statsmodels
用法：直接运行 `python arima_model.py` 查看 demo；或 import 后调用
      select_order_aic 自动定阶、arima_forecast 预测
"""
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 只抑制 statsmodels 拟合的收敛/值警告，不再全局静默（保留其他警告可见）
# 注意：必须在 import statsmodels 之后再设置，因为其导入时会重置 warnings 过滤器
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=ValueWarning)

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def select_order_aic(series, p_max=3, d_max=2, q_max=3):
    """网格搜索 AIC 最小的 ARIMA(p,d,q) 阶数组合

    参数:
        series: array_like, 一维时间序列
        p_max: int, AR 阶数搜索上限（含），默认 3
        d_max: int, 差分阶数搜索上限（含），默认 2
        q_max: int, MA 阶数搜索上限（含），默认 3
    返回:
        tuple: (best_order, best_aic)
            best_order 形如 (p, d, q)，best_aic 为对应的最小 AIC 值
    """
    series = np.asarray(series, dtype=float).ravel()
    best_aic = np.inf
    best_order = (0, 0, 0)
    # 遍历所有 (p, d, q) 组合，拟合失败或无法收敛的组合跳过
    for p in range(p_max + 1):
        for d in range(d_max + 1):
            for q in range(q_max + 1):
                if p == 0 and d == 0 and q == 0:
                    continue  # (0,0,0) 无意义，跳过
                try:
                    model = ARIMA(series, order=(p, d, q),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
                    res = model.fit()
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    print(f'自动定阶完成：最优阶数 (p,d,q) = {best_order}，AIC = {best_aic:.4f}')
    return best_order, best_aic


def arima_forecast(series, order, steps=5):
    """用指定阶数的 ARIMA 模型拟合并向前预测

    参数:
        series: array_like, 一维时间序列
        order: tuple, (p, d, q) 阶数
        steps: int, 向前预测步数，默认 5
    返回:
        dict:
            'forecast'         预测值（长度 steps 的 numpy 数组）
            'conf_int'         95% 置信区间，shape 为 (steps, 2)，列分别为下界/上界
            'model_summary_ic' 信息准则 dict {'aic','bic','hqic'}
    """
    series = np.asarray(series, dtype=float).ravel()
    # 与定阶搜索保持一致：不强制平稳/可逆约束，保证信息准则口径一致
    res = ARIMA(series, order=order,
                enforce_stationarity=False,
                enforce_invertibility=False).fit()
    pred = res.get_forecast(steps=steps)
    forecast = np.asarray(pred.predicted_mean)
    conf_int = np.asarray(pred.conf_int(alpha=0.05))  # 95% 置信区间
    return {
        'forecast': forecast,
        'conf_int': conf_int,
        'model_summary_ic': {
            'aic': float(res.aic),
            'bic': float(res.bic),
            'hqic': float(res.hqic),
        },
    }


if __name__ == '__main__':
    # ---------------- demo：带趋势的模拟序列，自动定阶 + 预测 ----------------
    np.random.seed(42)
    n = 80  # 序列长度取短一些，保证网格搜索在数秒内完成
    t = np.arange(n)
    # 线性趋势 + 周期波动 + AR(1) 型噪声（累积噪声使序列更接近真实数据）
    noise = np.cumsum(np.random.normal(0, 0.6, n))
    series = 10 + 0.35 * t + 3 * np.sin(2 * np.pi * t / 12) + noise

    print('=' * 60)
    print('ARIMA 自动定阶与预测 demo')
    print('=' * 60)

    # 第一步：AIC 网格搜索自动定阶（p_max=3, d_max=2, q_max=3）
    order, aic = select_order_aic(series, p_max=3, d_max=2, q_max=3)

    # 第二步：用最优阶数向前预测 5 步
    steps = 5
    result = arima_forecast(series, order, steps=steps)
    print(f"预测 {steps} 步: {np.round(result['forecast'], 4)}")
    print('95% 置信区间:')
    print(np.round(result['conf_int'], 4))
    ic = result['model_summary_ic']
    print(f"AIC = {ic['aic']:.4f}，BIC = {ic['bic']:.4f}，HQIC = {ic['hqic']:.4f}")

    # 第三步：画历史序列 + 预测值 + 置信区间
    t_future = np.arange(n, n + steps)
    plt.figure(figsize=(9, 5))
    plt.plot(t, series, 'o-', markersize=3, label='历史序列')
    plt.plot(np.concatenate([[n - 1], t_future]),
             np.concatenate([[series[-1]], result['forecast']]),
             'r^--', label='ARIMA 预测值')
    plt.fill_between(t_future, result['conf_int'][:, 0], result['conf_int'][:, 1],
                     color='red', alpha=0.2, label='95% 置信区间')
    plt.xlabel('时间 t')
    plt.ylabel('序列值')
    plt.title(f'ARIMA{order} 预测（AIC = {aic:.4f}）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('arima_demo.png', dpi=150, bbox_inches='tight')
    print('demo 运行结束，图片已保存为 arima_demo.png。')
