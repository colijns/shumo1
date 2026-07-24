"""
模块：grey_relation
功能：灰色关联分析（GRA）——计算各比较序列对参考序列的灰色关联度并排序
适用题型：国赛C题高频 / 华数杯高频（影响因素分析、综合评价、指标筛选）
依赖：numpy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python grey_relation.py` 查看 demo；或 import 后调用 grey_relational(X)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def grey_relational(X, ref_index=0, rho=0.5):
    """
    灰色关联分析：衡量各比较（子）序列与参考（母）序列的几何相似程度。

    计算步骤：
        1. 无量纲化：各序列除以其均值（均值化变换），消除量纲与数量级影响；
        2. 求差序列：Δ_ij = |x_i(j) - x_0(j)|；
        3. 两极差：d_min、d_max 为所有差值中的最小、最大值；
        4. 关联系数：ξ_ij = (d_min + ρ·d_max) / (Δ_ij + ρ·d_max)；
        5. 关联度：各时点关联系数的算术平均。

    参数：
        X         : numpy.ndarray，(m, n) 数据矩阵，每行一个序列、每列一个时点/样本；
                    第 ref_index 行为参考序列，其余行为比较序列
        ref_index : int，参考（母）序列所在行号，默认 0
        rho       : float，分辨系数，取值 (0, 1)，一般取 0.5
    返回：
        (m-1,) numpy.ndarray，各比较序列对参考序列的关联度（不含参考序列自身），
        顺序与 X 中剔除参考行后的行顺序一致；关联度越大关系越密切
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('X 必须是 (m 序列 × n 时点) 的二维矩阵')
    m, n = X.shape
    if m < 2:
        raise ValueError('至少需要 1 个参考序列和 1 个比较序列（m >= 2）')
    if not (0 < rho < 1):
        raise ValueError('分辨系数 rho 必须位于 (0, 1) 区间')

    # 1. 均值化无量纲处理
    row_mean = X.mean(axis=1, keepdims=True)
    row_mean = np.where(row_mean == 0, 1.0, row_mean)  # 防止除零
    Z = X / row_mean

    # 2. 参考序列与比较序列
    x0 = Z[ref_index]                       # 参考（母）序列
    others = np.delete(Z, ref_index, axis=0)  # 比较（子）序列

    # 3. 差序列与两极差
    diff = np.abs(others - x0)
    d_min, d_max = diff.min(), diff.max()

    # 4. 关联系数矩阵
    xi = (d_min + rho * d_max) / (diff + rho * d_max)

    # 5. 关联度：按时点取平均
    grade = xi.mean(axis=1)
    return grade


if __name__ == '__main__':
    np.random.seed(42)

    # ===== demo：某市 GDP 与 4 个影响因素的灰色关联分析（6 年数据）=====
    years = np.arange(2018, 2024)
    # 母序列：地区生产总值 GDP（亿元）
    gdp = np.array([3200, 3550, 3480, 3900, 4250, 4700], dtype=float)
    # 子序列：4 个影响因素
    invest = np.array([980, 1120, 1050, 1230, 1380, 1550], dtype=float)   # 固定资产投资
    finance = np.array([410, 465, 430, 520, 570, 640], dtype=float)       # 财政收入
    retail = np.array([1250, 1380, 1210, 1520, 1660, 1830], dtype=float)  # 社会消费品零售总额
    students = np.array([52, 54, 55, 56, 57, 58], dtype=float)            # 在校大学生数（万人）

    factor_names = ['固定资产投资', '财政收入', '社会消费品零售总额', '在校大学生数']
    X = np.vstack([gdp, invest, finance, retail, students])  # 第 0 行为母序列

    print('===== 灰色关联分析 demo：GDP 影响因素关联度排序 =====')
    print(f'年份：{[int(y) for y in years]}')
    print(f'母序列（GDP）：{gdp}')
    for name, row in zip(factor_names, X[1:]):
        print(f'子序列（{name}）：{row}')

    grades = grey_relational(X, ref_index=0, rho=0.5)

    # ---- 按关联度从大到小排序输出 ----
    order = np.argsort(grades)[::-1]
    print('\n----- 各因素对 GDP 的灰色关联度（降序）-----')
    for rank, idx in enumerate(order, start=1):
        print(f'第 {rank} 名：{factor_names[idx]}，关联度 = {grades[idx]:.4f}')

    # ---- 可视化：关联度柱状图 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    sorted_names = [factor_names[i] for i in order]
    sorted_grades = grades[order]
    ax.bar(sorted_names, sorted_grades, color='#4C72B0', edgecolor='black')
    for i, g in enumerate(sorted_grades):
        ax.text(i, g + 0.005, f'{g:.4f}', ha='center', fontsize=10)
    ax.set_title('各影响因素对 GDP 的灰色关联度')
    ax.set_ylabel('灰色关联度')
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig('grey_relation_demo.png', dpi=150, bbox_inches='tight')
    print('grey_relation_demo.png 已保存')
    plt.close(fig)
    print('\ngrey_relation demo 运行完毕')
