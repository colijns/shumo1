"""
模块：fuzzy_evaluation
功能：模糊综合评价——由单因素模糊评价矩阵与权重向量合成综合隶属度，按最大隶属度原则给出结论
适用题型：国赛C题高频 / 华数杯高频（含主观评价、等级评定类问题）
依赖：numpy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python fuzzy_evaluation.py` 查看 demo；或 import 后调用 fuzzy_evaluate(R, w)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def fuzzy_evaluate(R, w, operator='minmax'):
    """
    模糊综合评价：B = w ∘ R，得到对待评对象各评语等级的综合隶属度。

    参数：
        R        : numpy.ndarray，(m, n) 单因素模糊评价矩阵，
                   R[i, j] 表示第 i 个评价因素对第 j 个评语等级的隶属度，
                   通常每行归一化（行和为 1）
        w        : array_like，(m,) 各因素权重（函数内部自动归一化）
        operator : str，模糊合成算子：
                   'minmax'   M(∧, ∨)：先取小后取大，突出主因素（默认）
                   'multiply' M(·, +)：加权求和，保留全部信息
    返回：
        (n,) numpy.ndarray，综合隶属度向量 B，B[j] 越大说明越倾向于第 j 个等级
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError('R 必须是 (m 因素 × n 等级) 的二维矩阵')
    m, n = R.shape
    w = np.asarray(w, dtype=float).ravel()
    if w.shape[0] != m:
        raise ValueError(f'权重长度（{w.shape[0]}）必须与因素数（{m}）一致')
    w = w / w.sum()  # 权重归一化

    if operator == 'minmax':
        # M(∧, ∨)：B_j = max_i min(w_i, R_ij)
        B = np.zeros(n)
        for j in range(n):
            B[j] = np.max(np.minimum(w, R[:, j]))
    elif operator == 'multiply':
        # M(·, +)：B_j = Σ_i w_i * R_ij（矩阵乘法）
        B = w @ R
    else:
        raise ValueError("operator 仅支持 'minmax'（M(∧,∨)）或 'multiply'（M(·,+)）")
    return B


if __name__ == '__main__':
    # ===== demo：教师课堂教学质量模糊综合评价（4 因素 × 5 等级）=====
    factors = ['教学内容', '教学方法', '教学态度', '教学效果']
    grades = ['优秀', '良好', '中等', '及格', '差']

    # 单因素模糊评价矩阵：由学生投票比例得到，每行和为 1
    R = np.array([
        [0.50, 0.30, 0.20, 0.00, 0.00],   # 教学内容
        [0.30, 0.40, 0.20, 0.10, 0.00],   # 教学方法
        [0.40, 0.40, 0.10, 0.10, 0.00],   # 教学态度
        [0.20, 0.30, 0.30, 0.10, 0.10],   # 教学效果
    ])
    w = np.array([0.30, 0.25, 0.20, 0.25])  # 4 个因素的权重

    print('===== 模糊综合评价 demo：教师课堂教学评价 =====')
    print(f'评价因素：{factors}')
    print(f'评语等级：{grades}')
    print(f'因素权重 w = {np.round(w, 4)}')
    print('单因素评价矩阵 R =')
    print(np.round(R, 4))

    # ---- 两种合成算子分别计算 ----
    results = {}
    for op, op_name in [('minmax', 'M(∧,∨) 取小取大'), ('multiply', 'M(·,+) 加权求和')]:
        B = fuzzy_evaluate(R, w, operator=op)
        results[op] = B
        best = int(np.argmax(B))
        print(f'\n----- 算子 {op_name} -----')
        for g, bj in zip(grades, B):
            print(f'  {g}：隶属度 = {bj:.4f}')
        print(f'按最大隶属度原则，该教师教学评价结论为：【{grades[best]}】'
              f'（隶属度 {B[best]:.4f}）')

    # ---- 可视化：两种算子的隶属度分布对比 ----
    x = np.arange(len(grades))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, results['minmax'], width, label='M(∧,∨)',
           color='#4C72B0', edgecolor='black')
    ax.bar(x + width / 2, results['multiply'], width, label='M(·,+)',
           color='#DD8452', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.set_ylabel('综合隶属度')
    ax.set_title('模糊综合评价结果（两种合成算子对比）')
    ax.legend()
    plt.tight_layout()
    fig.savefig('fuzzy_evaluation_demo.png', dpi=150, bbox_inches='tight')
    print('fuzzy_evaluation_demo.png 已保存')
    plt.close(fig)
    print('\nfuzzy_evaluation demo 运行完毕')
