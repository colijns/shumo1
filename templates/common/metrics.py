"""
模块：metrics
功能：常用预测精度评价指标：RMSE（均方根误差）、MAE（平均绝对误差）、MAPE（平均绝对百分比误差）、R²（决定系数）
适用题型：通用（预测类题目结果检验必备）
依赖：numpy
用法：直接运行 `python metrics.py` 查看 demo；或 import 后调用各指标函数
"""

import numpy as np


def _check_pair(y_true, y_pred):
    """内部辅助：把输入转成等长的一维 float 数组并做长度校验。"""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f'y_true 与 y_pred 长度不一致：{y_true.shape} vs {y_pred.shape}')
    return y_true, y_pred


def rmse(y_true, y_pred):
    """
    均方根误差 RMSE = sqrt(mean((y_true - y_pred)^2))，越小越好，对异常值敏感。

    参数：
        y_true : array_like，真实值序列
        y_pred : array_like，预测值序列
    返回：
        float，RMSE 值
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """
    平均绝对误差 MAE = mean(|y_true - y_pred|)，越小越好。

    参数：
        y_true : array_like，真实值序列
        y_pred : array_like，预测值序列
    返回：
        float，MAE 值
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    """
    平均绝对百分比误差 MAPE = mean(|(y_true - y_pred) / y_true|) × 100（单位：%），
    越小越好；真实值为 0 的样本会被自动剔除以避免除零。

    参数：
        y_true : array_like，真实值序列
        y_pred : array_like，预测值序列
    返回：
        float，MAPE 值（百分数）
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    mask = y_true != 0
    if not np.any(mask):
        raise ValueError('y_true 全为 0，无法计算 MAPE')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def r2_score(y_true, y_pred):
    """
    决定系数 R² = 1 - SS_res / SS_tot，越接近 1 拟合越好，可为负（比均值预测还差）。

    参数：
        y_true : array_like，真实值序列
        y_pred : array_like，预测值序列
    返回：
        float，R² 值
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)      # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
    if ss_tot == 0:
        raise ValueError('y_true 为常数序列，R² 无定义')
    return float(1.0 - ss_res / ss_tot)


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：构造真实值与带噪声的预测值，打印四个指标 =====
    n = 20
    y_true = np.linspace(10, 100, n) + np.random.randn(n) * 3.0   # 模拟真实值
    y_pred = y_true + np.random.randn(n) * 5.0                    # 模拟有偏噪声预测

    print('===== 预测精度评价指标 =====')
    print(f'样本数 n   = {n}')
    print(f'RMSE       = {rmse(y_true, y_pred):.4f}')
    print(f'MAE        = {mae(y_true, y_pred):.4f}')
    print(f'MAPE       = {mape(y_true, y_pred):.4f} %')
    print(f'R²         = {r2_score(y_true, y_pred):.4f}')
    print('\nmetrics demo 运行完毕')
