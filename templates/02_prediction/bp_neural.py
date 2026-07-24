# -*- coding: utf-8 -*-
"""
模块：bp_neural
功能：BP 神经网络时间序列预测（sklearn MLPRegressor，滑动窗口构造样本，含测试集评估与递归外推）
适用题型：通用高频（非线性时间序列预测，国赛C题/华数杯常用，可与传统方法对比）
依赖：numpy, matplotlib, sklearn
用法：直接运行 `python bp_neural.py` 查看 demo；或 import 后调用 bp_forecast
"""
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

# 抑制 MLPRegressor 未完全收敛的警告（demo 中迭代次数有限，属正常现象）
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


# ---------------- 内部自实现的评价指标（不跨目录 import，保证单文件独立） ----------------
def _rmse(y_true, y_pred):
    """均方根误差 RMSE"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    """平均绝对误差 MAE"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _mape(y_true, y_pred):
    """平均绝对百分比误差 MAPE（%），自动剔除真值为 0 的样本"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _r2_score(y_true, y_pred):
    """决定系数 R²"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def _make_supervised(series, lag):
    """用滑动窗口把一维时间序列构造为监督学习样本

    参数:
        series: ndarray, 一维序列
        lag: int, 滞后步数（用前 lag 个点预测下一个点）
    返回:
        (X, y): X 形状 (n_samples, lag)，y 形状 (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - lag):
        X.append(series[i:i + lag])
        y.append(series[i + lag])
    return np.array(X), np.array(y)


def bp_forecast(series, lag=5, test_ratio=0.2, steps=0, seed=None):
    """BP 神经网络时间序列预测

    流程：序列标准化 -> 滑动窗口构造样本 -> 按时间顺序划分训练/测试集
         -> MLPRegressor 训练 -> 测试集评估 -> 递归多步外推未来值

    参数:
        series: array_like, 一维时间序列
        lag: int, 滞后步数（用前 lag 个点预测下一个点），默认 5
        test_ratio: float, 测试集占样本比例（时间尾部），默认 0.2
        steps: int, 训练结束后递归外推的未来步数，0 表示不外推
        seed: int 或 None, 随机种子，设为整数时保证可复现（MLPRegressor random_state），默认 None
    返回:
        dict:
            'y_pred_test' 测试集预测值（numpy 数组）
            'y_true_test' 测试集真实值（numpy 数组，便于对比画图）
            'future'      递归外推的未来 steps 步预测值（steps=0 时为空数组）
            'metrics'     测试集指标 dict {'rmse','mae','mape','r2'}
            'model'       训练好的 MLPRegressor 对象
    """
    series = np.asarray(series, dtype=float).ravel()
    if seed is not None:
        np.random.seed(seed)
    # 标准化（神经网络对量纲敏感，能显著改善收敛）
    mu, sigma = series.mean(), series.std()
    if sigma == 0:
        sigma = 1.0
    series_std = (series - mu) / sigma

    # 滑动窗口构造监督学习样本
    X, y = _make_supervised(series_std, lag)
    n_samples = len(X)
    n_test = max(1, int(n_samples * test_ratio))
    n_train = n_samples - n_test
    # 时间序列必须按时间顺序划分，不能随机打乱
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # 训练 BP 神经网络（单隐层 100 节点，adam 优化器）
    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                         solver='adam', max_iter=2000,
                         random_state=seed if seed is not None else None)
    model.fit(X_train, y_train)

    # 测试集预测并反标准化回原量纲
    y_pred_std = model.predict(X_test)
    y_pred = y_pred_std * sigma + mu
    y_true = y_test * sigma + mu

    metrics = {
        'rmse': _rmse(y_true, y_pred),
        'mae': _mae(y_true, y_pred),
        'mape': _mape(y_true, y_pred),
        'r2': _r2_score(y_true, y_pred),
    }

    # 递归多步外推：每次用最近 lag 个点预测下一点，并把预测值并入窗口
    future = []
    if steps > 0:
        window = series_std[-lag:].copy()
        for _ in range(steps):
            next_std = model.predict(window.reshape(1, -1))[0]
            future.append(next_std * sigma + mu)
            window = np.append(window[1:], next_std)  # 窗口向前滑动
    future = np.array(future)

    return {
        'y_pred_test': y_pred,
        'y_true_test': y_true,
        'future': future,
        'metrics': metrics,
        'model': model,
    }


if __name__ == '__main__':
    # ---------------- demo：模拟非线性序列，滞后 5 步建模 ----------------
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    # 趋势 + 双周期叠加 + 噪声，具有一定非线性特征
    series = 0.05 * t + 5 * np.sin(2 * np.pi * t / 24) \
        + 2 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 0.5, n)

    lag = 5
    steps = 10
    print('=' * 60)
    print(f'BP 神经网络时间序列预测 demo（滞后 {lag} 步，外推 {steps} 步）')
    print('=' * 60)
    result = bp_forecast(series, lag=lag, test_ratio=0.2, steps=steps, seed=42)

    m = result['metrics']
    print(f"测试集指标：RMSE = {m['rmse']:.4f}，MAE = {m['mae']:.4f}，"
          f"MAPE = {m['mape']:.4f}%，R^2 = {m['r2']:.4f}")
    print('未来 10 步外推预测值:', np.round(result['future'], 4))

    # 画测试集预测对比图 + 未来外推
    n_test = len(result['y_true_test'])
    t_test = np.arange(n - n_test, n)
    t_future = np.arange(n, n + steps)
    plt.figure(figsize=(10, 5))
    plt.plot(t, series, color='steelblue', alpha=0.7, label='原始序列')
    plt.plot(t_test, result['y_pred_test'], 'r--', label='测试集预测值')
    if steps > 0:
        plt.plot(np.concatenate([[n - 1], t_future]),
                 np.concatenate([[series[-1]], result['future']]),
                 'g^--', label='未来外推值')
    plt.xlabel('时间 t')
    plt.ylabel('序列值')
    plt.title(f"BP 神经网络时间序列预测（R^2 = {m['r2']:.4f}）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('bp_neural_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('demo 运行结束。')
