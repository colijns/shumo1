# -*- coding: utf-8 -*-
"""
模块：hypothesis_tests
功能：6 种常用假设检验——正态性检验、单样本 t、双样本 t、配对 t、单因素方差分析、卡方独立性检验
适用题型：通用（数据差异性/分布检验，国赛与华数杯统计分析部分常用）
依赖：numpy, scipy
用法：直接运行 `python hypothesis_tests.py` 查看 demo；或 import 后调用各检验函数
      所有函数均返回 (统计量, p值)，并在显著性水平 α=0.05 下打印中文结论
"""
import numpy as np
from scipy import stats

ALPHA = 0.05  # 默认显著性水平


def _conclusion(p, h0_desc, alpha=ALPHA):
    """按 p 值与显著性水平打印中文结论

    参数:
        p: float, 检验 p 值
        h0_desc: str, 原假设的中文描述
        alpha: float, 显著性水平，默认 0.05
    """
    if p < alpha:
        print(f'结论：p = {p:.4f} < {alpha}，拒绝原假设（{h0_desc}），差异/关系具有统计显著性。')
    else:
        print(f'结论：p = {p:.4f} >= {alpha}，不能拒绝原假设（{h0_desc}）。')


def normality_test(x):
    """正态性检验（Shapiro-Wilk）

    原假设 H0：样本来自正态分布总体（p >= 0.05 时认为服从正态分布）

    参数:
        x: array_like, 一维样本
    返回:
        tuple: (统计量 W, p值)
    """
    x = np.asarray(x, dtype=float).ravel()
    stat, p = stats.shapiro(x)
    print(f'【正态性检验 Shapiro-Wilk】W = {stat:.4f}，p = {p:.4f}')
    _conclusion(p, '样本来自正态分布总体')
    return stat, p


def one_sample_t(x, mu):
    """单样本 t 检验

    原假设 H0：总体均值等于 mu

    参数:
        x: array_like, 一维样本
        mu: float, 待检验的总体均值
    返回:
        tuple: (统计量 t, p值)
    """
    x = np.asarray(x, dtype=float).ravel()
    stat, p = stats.ttest_1samp(x, mu)
    print(f'【单样本 t 检验】t = {stat:.4f}，p = {p:.4f}（样本均值 = {x.mean():.4f}，'
          f'待检验均值 = {mu}）')
    _conclusion(p, f'总体均值等于 {mu}')
    return stat, p


def two_sample_t(x, y, equal_var=False):
    """两独立样本 t 检验

    原假设 H0：两总体均值相等
    默认 equal_var=False（Welch t，不假定方差齐性，更稳健，推荐）；
    若确信方差齐性，可传 equal_var=True 切回 Student t。

    参数:
        x, y: array_like, 两组独立样本
        equal_var: bool, 是否假定方差齐性，默认 False（Welch）
    返回:
        tuple: (统计量 t, p值)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    stat, p = stats.ttest_ind(x, y, equal_var=equal_var)
    print(f'【两独立样本 t 检验（{"Student" if equal_var else "Welch"}）】t = {stat:.4f}，p = {p:.4f}'
          f'（均值1 = {x.mean():.4f}，均值2 = {y.mean():.4f}）')
    _conclusion(p, '两总体均值相等')
    return stat, p


def paired_t(x, y):
    """配对样本 t 检验

    原假设 H0：成对差值的总体均值为 0（即处理前后无差异）

    参数:
        x, y: array_like, 两组配对样本（长度相同，如同一批人处理前后）
    返回:
        tuple: (统计量 t, p值)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    stat, p = stats.ttest_rel(x, y)
    print(f'【配对样本 t 检验】t = {stat:.4f}，p = {p:.4f}'
          f'（差值均值 = {np.mean(x - y):.4f}）')
    _conclusion(p, '成对差值的总体均值为 0')
    return stat, p


def one_way_anova(*groups):
    """单因素方差分析（one-way ANOVA）

    原假设 H0：各组总体均值全部相等

    参数:
        *groups: 可变参数，每个参数为一组样本（至少 2 组）
    返回:
        tuple: (统计量 F, p值)
    """
    if len(groups) < 2:
        raise ValueError('方差分析至少需要 2 组样本')
    groups = [np.asarray(g, dtype=float).ravel() for g in groups]
    stat, p = stats.f_oneway(*groups)
    means_str = '，'.join(f'组{i + 1}均值 = {g.mean():.4f}'
                          for i, g in enumerate(groups))
    print(f'【单因素方差分析】F = {stat:.4f}，p = {p:.4f}（{means_str}）')
    _conclusion(p, '各组总体均值全部相等')
    return stat, p


def chi2_independence(table):
    """卡方独立性检验（列联表）

    原假设 H0：行变量与列变量相互独立

    参数:
        table: array_like, r×c 的观测频数列联表
    返回:
        tuple: (统计量 chi2, p值)
    """
    table = np.asarray(table, dtype=float)
    chi2, p, dof, expected = stats.chi2_contingency(table)
    print(f'【卡方独立性检验】chi2 = {chi2:.4f}，p = {p:.4f}，自由度 = {dof}')
    print('期望频数表:')
    print(np.round(expected, 4))
    _conclusion(p, '行变量与列变量相互独立')
    return chi2, p


if __name__ == '__main__':
    np.random.seed(42)
    print('=' * 60)
    print('常用假设检验 demo（显著性水平 α = 0.05）')
    print('=' * 60)

    # 案例 1：正态性检验——正态样本应不能拒绝 H0
    print('\n--- 1. 正态性检验 ---')
    x_norm = np.random.normal(100, 15, 100)
    normality_test(x_norm)

    # 案例 2：单样本 t 检验——某车间零件直径标称 10mm
    print('\n--- 2. 单样本 t 检验 ---')
    diameter = np.random.normal(10.02, 0.05, 30)
    one_sample_t(diameter, mu=10.0)

    # 案例 3：两独立样本 t 检验——两种工艺的产品强度比较
    print('\n--- 3. 两独立样本 t 检验 ---')
    process_a = np.random.normal(50, 5, 40)
    process_b = np.random.normal(52, 5, 40)
    two_sample_t(process_a, process_b)

    # 案例 4：配对样本 t 检验——同一批患者服药前后血压比较
    print('\n--- 4. 配对样本 t 检验 ---')
    before = np.random.normal(140, 10, 25)
    after = before - np.random.normal(8, 5, 25)  # 服药后平均下降约 8
    paired_t(before, after)

    # 案例 5：单因素方差分析——三种教学方法的成绩比较
    print('\n--- 5. 单因素方差分析 ---')
    group1 = np.random.normal(80, 8, 30)
    group2 = np.random.normal(82, 8, 30)
    group3 = np.random.normal(86, 8, 30)
    one_way_anova(group1, group2, group3)

    # 案例 6：卡方独立性检验——性别与是否偏好某产品
    print('\n--- 6. 卡方独立性检验 ---')
    table = np.array([[60, 40],   # 男性：偏好 / 不偏好
                      [45, 55]])  # 女性：偏好 / 不偏好
    chi2_independence(table)

    print('\ndemo 运行结束。')
