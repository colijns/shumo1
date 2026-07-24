"""
模块：data_utils
功能：数据预处理工具箱：极差归一化、Z-score 标准化、指标正向化（效益/成本/区间/中间型）、缺失值填充
适用题型：通用（评价类、统计类题目的数据预处理环节）
依赖：numpy, pandas
用法：直接运行 `python data_utils.py` 查看 demo；或 import 后调用各预处理函数
"""

import numpy as np
import pandas as pd


def min_max_normalize(X, axis=0):
    """
    极差归一化，把数据线性缩放到 [0, 1] 区间：x' = (x - min) / (max - min)。

    参数：
        X    : numpy.ndarray，待归一化数据（一维或二维）
        axis : int，沿哪一轴求最值，默认 0（按列，即每个指标单独归一化）
    返回：
        numpy.ndarray，与 X 同形状的归一化结果；极差为 0 的列（取值全相同）返回全 0
    """
    X = np.asarray(X, dtype=float)
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)
    span = x_max - x_min
    span = np.where(span == 0, 1.0, span)  # 防止除零：常数列无法区分优劣，结果为 0
    return (X - x_min) / span


def z_score_standardize(X, axis=0):
    """
    Z-score 标准化，使数据沿指定轴均值为 0、标准差为 1：x' = (x - μ) / σ。

    参数：
        X    : numpy.ndarray，待标准化数据（一维或二维）
        axis : int，沿哪一轴统计，默认 0（按列）
    返回：
        numpy.ndarray，标准化结果；标准差为 0 的列返回全 0
    """
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=axis, keepdims=True)
    sigma = np.std(X, axis=axis, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)  # 防止除零
    return (X - mu) / sigma


def _take_param(param, j, name):
    """内部辅助：按列号 j 取参数值，同时支持 {列号: 值} 字典与按列对齐的列表两种传法。"""
    if param is None:
        raise ValueError(f'第 {j} 个指标需要提供 {name} 参数，但当前为 None')
    return param[j]


def forward_transform(X, types, bounds=None, best=None):
    """
    指标正向化：把成本型、区间型、中间型指标统一转化为效益型（越大越好）。

    参数：
        X      : numpy.ndarray 或 pandas.DataFrame，(n 方案 × m 指标) 原始数据，也支持一维
        types  : list[str]，长度 m，每个元素为指标类型：
                 'benefit'  效益型（越大越好，保持原值不变）
                 'cost'     成本型（越小越好，用 max - x 转化）
                 'interval' 区间型（落在最优区间 [a, b] 内最好，必须配合 bounds）
                 'mid'      中间型（越接近最优值 t 越好，必须配合 best）
        bounds : dict 或 list，区间型指标的最优区间，如 {2: (6, 8)} 或按列对齐的 [(a, b), ...]
        best   : dict 或 list，中间型指标的最优值，如 {3: 90} 或按列对齐的 [t, ...]
    返回：
        numpy.ndarray，正向化后的二维数据（所有指标均为越大越好，取值落在 [0, 1] 或原值域内）
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = X.copy()
    n, m = X.shape
    if len(types) != m:
        raise ValueError(f'types 长度（{len(types)}）必须与指标数（{m}）一致')

    for j in range(m):
        t = types[j]
        col = X[:, j]
        if t == 'benefit':
            pass  # 效益型不变
        elif t == 'cost':
            # 成本型：max - x，原始值越小转化后越大
            X[:, j] = col.max() - col
        elif t == 'interval':
            # 区间型：经典教科书公式，区间内取 1，区间外按偏离距离线性衰减
            a, b = _take_param(bounds, j, 'bounds')
            M = max(a - col.min(), col.max() - b)  # 最大偏离量
            if M <= 0:
                X[:, j] = 1.0  # 所有值都落在最优区间内
            else:
                X[:, j] = np.where(col < a, 1 - (a - col) / M,
                                   np.where(col > b, 1 - (col - b) / M, 1.0))
        elif t == 'mid':
            # 中间型：1 - |x - t| / max|x - t|，越接近最优值越大
            t0 = _take_param(best, j, 'best')
            M = np.max(np.abs(col - t0))
            X[:, j] = 1.0 if M == 0 else 1 - np.abs(col - t0) / M
        else:
            raise ValueError(f"未知指标类型 '{t}'，仅支持 'benefit'/'cost'/'interval'/'mid'")
    return X


def fill_missing(df, method='mean'):
    """
    缺失值填充。

    参数：
        df     : pandas.DataFrame，可能含缺失值（NaN）的数据框
        method : str，填充方式：
                 'mean'        各列均值填充
                 'median'      各列中位数填充
                 'ffill'       前向填充（用上一行值，开头缺失再后向补齐）
                 'interpolate' 线性插值填充（首尾缺失再前/后向补齐）
    返回：
        pandas.DataFrame，填充后的新数据框（不修改原 df）
    """
    df = df.copy()
    if method == 'mean':
        return df.fillna(df.mean(numeric_only=True))
    elif method == 'median':
        return df.fillna(df.median(numeric_only=True))
    elif method == 'ffill':
        return df.ffill().bfill()
    elif method == 'interpolate':
        return df.interpolate().bfill().ffill()
    else:
        raise ValueError("method 仅支持 'mean'、'median'、'ffill'、'interpolate'")


if __name__ == '__main__':
    np.random.seed(42)

    # ===== 构造含缺失值、且含成本型/区间型/中间型指标的示例数据框 =====
    df = pd.DataFrame({
        '疗效(效益型)': [80.0, 92.0, np.nan, 88.0, 76.0],
        '费用(成本型)': [5000.0, 4200.0, 6100.0, np.nan, 5800.0],
        '住院天数(区间型)': [9.0, 7.0, 12.0, 6.0, 10.0],
        '血药浓度(中间型)': [90.0, 85.0, 70.0, 95.0, 60.0],
    })
    print('===== 1. 原始数据（含缺失值）=====')
    print(df)

    # ===== 2. 缺失值填充（演示四种方法）=====
    print('\n===== 2. 缺失值填充结果对比 =====')
    for mtd in ['mean', 'median', 'ffill', 'interpolate']:
        filled = fill_missing(df, method=mtd)
        print(f"\n-- method='{mtd}' --")
        print(filled.round(4))

    # 后续流程使用均值填充结果
    df_filled = fill_missing(df, method='mean')

    # ===== 3. 指标正向化 =====
    # 区间型：住院天数最优区间为 [6, 8]；中间型：血药浓度最优值为 90
    X_pos = forward_transform(df_filled.values,
                              types=['benefit', 'cost', 'interval', 'mid'],
                              bounds={2: (6, 8)},
                              best={3: 90})
    print('\n===== 3. 正向化后（全部越大越好）=====')
    print(pd.DataFrame(X_pos, columns=df.columns).round(4))

    # ===== 4. 归一化与标准化 =====
    print('\n===== 4. 极差归一化（[0,1]）=====')
    print(pd.DataFrame(min_max_normalize(X_pos, axis=0), columns=df.columns).round(4))
    print('\n===== 5. Z-score 标准化（均值0 方差1）=====')
    print(pd.DataFrame(z_score_standardize(X_pos, axis=0), columns=df.columns).round(4))
    print('\ndata_utils demo 运行完毕')
