# -*- coding: utf-8 -*-
"""
模块：grey_model
功能：GM(1,1) 灰色预测模型，含后验差检验（C 后验差比、P 小误差概率）与精度等级判定
适用题型：通用高频（小样本、单调趋势类时间序列预测，国赛C题/华数杯常用）
依赖：numpy, matplotlib
用法：直接运行 `python grey_model.py` 查看 demo；或 import 后调用核心函数 gm11
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体与负号显示（保证单文件独立运行）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def _judge_grade(C, P):
    """根据后验差比 C 与小误差概率 P 判定模型精度等级（查表法）

    参数:
        C: float, 后验差比值
        P: float, 小误差概率
    返回:
        str: 精度等级中文描述
    """
    if P > 0.95 and C < 0.35:
        return '好（一级）'
    elif P > 0.80 and C < 0.50:
        return '合格（二级）'
    elif P > 0.70 and C < 0.65:
        return '勉强合格（三级）'
    else:
        return '不合格（四级）'


def gm11(x0, n_predict=3):
    """GM(1,1) 灰色预测模型

    建模步骤：一次累加生成(1-AGO) -> 紧邻均值生成 -> 最小二乘估计发展系数 a 与灰作用量 b
             -> 白化方程时间响应式 -> 累减还原 -> 后验差检验

    参数:
        x0: array_like, 原始非负序列，长度建议 >= 4（小样本建模）
        n_predict: int, 向后预测的期数，默认 3
    返回:
        dict:
            'a'             发展系数
            'b'             灰作用量
            'fitted'        历史数据的拟合值（累减还原后，长度与 x0 相同）
            'predict'       未来 n_predict 期预测值
            'relative_error' 各期相对误差（%）
            'C'             后验差比（残差标准差 / 原序列标准差）
            'P'             小误差概率
            'grade'         精度等级中文描述
    """
    x0 = np.asarray(x0, dtype=float).ravel()
    n = len(x0)
    if n < 4:
        raise ValueError('GM(1,1) 建议序列长度至少为 4')
    if np.any(x0 < 0):
        raise ValueError('GM(1,1) 要求原始序列非负，请先平移处理')

    # 第一步：一次累加生成序列（1-AGO），弱化原始序列的随机性
    x1 = np.cumsum(x0)

    # 第二步：紧邻均值生成序列 z1(k) = 0.5*x1(k) + 0.5*x1(k-1)
    z1 = 0.5 * (x1[1:] + x1[:-1])

    # 第三步：构造数据矩阵 B 与数据向量 Y，最小二乘估计参数 [a, b]
    B = np.column_stack([-z1, np.ones(n - 1)])
    Y = x0[1:]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # 第四步：白化方程 dx1/dt + a*x1 = b 的时间响应函数
    def x1_hat(k):
        """预测第 k 期（k 从 0 开始）的累加值"""
        return (x0[0] - b / a) * np.exp(-a * k) + b / a

    # 累加序列的模拟/预测值（k = 0 .. n+n_predict-1）
    k_all = np.arange(n + n_predict)
    x1_all = np.array([x1_hat(k) for k in k_all])

    # 第五步：累减还原得到原始序列的拟合值与预测值
    x0_all = np.empty(n + n_predict)
    x0_all[0] = x0[0]  # 首期的拟合值约定为原始值
    x0_all[1:] = np.diff(x1_all)
    fitted = x0_all[:n]
    predict = x0_all[n:]

    # 第六步：精度检验
    residual = x0 - fitted                              # 残差
    relative_error = np.abs(residual) / x0 * 100        # 相对误差（%）
    # 后验差检验：C 越小越好，P 越大越好
    s1 = np.std(x0, ddof=1)                             # 原序列标准差
    s2 = np.std(residual, ddof=1)                       # 残差标准差
    C = s2 / s1 if s1 > 0 else np.inf
    # 小误差概率：残差偏离其均值不超过 0.6745*s1 的样本占比
    P = float(np.mean(np.abs(residual - np.mean(residual)) < 0.6745 * s1))
    grade = _judge_grade(C, P)

    return {
        'a': float(a),
        'b': float(b),
        'fitted': fitted,
        'predict': predict,
        'relative_error': relative_error,
        'C': float(C),
        'P': P,
        'grade': grade,
    }


if __name__ == '__main__':
    # ---------------- demo：8 期历史数据预测未来 3 期 ----------------
    # 某产品近 8 期销量（小样本、近似指数增长趋势，适合灰色预测）
    x0 = np.array([2.67, 3.13, 3.25, 3.36, 3.56, 3.72, 3.95, 4.12])
    n_predict = 3

    print('=' * 60)
    print('GM(1,1) 灰色预测 demo：8 期历史数据，预测未来 3 期')
    print('=' * 60)
    result = gm11(x0, n_predict=n_predict)

    print(f"发展系数 a = {result['a']:.4f}，灰作用量 b = {result['b']:.4f}")
    print('历史值  :', np.round(x0, 4))
    print('拟合值  :', np.round(result['fitted'], 4))
    print('相对误差:', np.round(result['relative_error'], 4), '(%)')
    print('预测值  :', np.round(result['predict'], 4))
    print(f"后验差比 C = {result['C']:.4f}，小误差概率 P = {result['P']:.4f}")
    print(f"模型精度等级：{result['grade']}")

    # 画历史值-拟合值-预测值对比图
    n = len(x0)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n + 1), x0, 'o-', label='历史数据')
    plt.plot(range(1, n + 1), result['fitted'], 's--', label='GM(1,1) 拟合值')
    plt.plot(range(n, n + n_predict + 1),
             np.concatenate([[x0[-1]], result['predict']]),
             '^--', color='red', label='预测值')
    plt.xlabel('时期')
    plt.ylabel('数值')
    plt.title(f"GM(1,1) 灰色预测（精度等级：{result['grade']}）")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('grey_model_demo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('demo 运行结束。')
