"""
模块：ahp
功能：层次分析法（AHP）——特征值法求判断矩阵权重，并进行一致性检验（CI / CR）
适用题型：国赛C题高频 / 华数杯高频（主观赋权、方案综合评价）
依赖：numpy, matplotlib（仅 demo 绘图用）
用法：直接运行 `python ahp.py` 查看 demo；或 import 后调用 ahp(A)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 随机一致性指标 RI（Saaty 随机模拟给出，key 为判断矩阵阶数 n）
RI_DICT = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
           6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp(A):
    """
    层次分析法：特征值法计算权重并做一致性检验。

    参数：
        A : numpy.ndarray，n×n 正互反判断矩阵，要求 a_ij > 0 且 a_ij = 1 / a_ji，
            a_ij 用 1~9 标度表示因素 i 相对因素 j 的重要程度
    返回：
        dict：
            weights    : (n,) numpy.ndarray，归一化权重向量（和为 1）
            lambda_max : float，判断矩阵最大特征值
            CI         : float，一致性指标 CI = (λmax - n) / (n - 1)
            CR         : float，一致性比率 CR = CI / RI
            consistent : bool，CR < 0.1 视为通过一致性检验
    """
    A = np.asarray(A, dtype=float)
    # ---- 输入校验 ----
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('判断矩阵必须是二维方阵')
    n = A.shape[0]
    if n < 1:
        raise ValueError('判断矩阵不能为空')
    if np.any(A <= 0):
        raise ValueError('判断矩阵元素必须全部为正数')
    if not np.allclose(A, 1.0 / A.T, atol=1e-6):
        print('警告：判断矩阵不满足互反性 a_ij = 1/a_ji，结果仅供参考')

    # ---- 特征值法求权重 ----
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmax(eigvals.real))          # 最大特征值下标
    lambda_max = float(eigvals[idx].real)
    w = eigvecs[:, idx].real                    # 对应特征向量（Perron 向量）
    w = np.abs(w)                               # 防止符号整体翻转
    weights = w / w.sum()                       # 归一化

    # ---- 一致性检验 ----
    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_DICT.get(n, 1.49)                   # n > 10 时近似取 1.49
    CR = CI / RI if RI > 0 else 0.0             # n <= 2 时恒一致，CR 记 0

    return {'weights': weights,
            'lambda_max': lambda_max,
            'CI': float(CI),
            'CR': float(CR),
            'consistent': bool(CR < 0.1)}


if __name__ == '__main__':
    # ===== demo：经典「选择旅游地」问题——准则层对目标层的判断矩阵 =====
    # 4 个准则：景色、费用、居住、饮食（1~9 标度互反矩阵）
    names = ['景色', '费用', '居住', '饮食']
    A = np.array([
        [1,    1/2,  4,    3  ],
        [2,    1,    7,    5  ],
        [1/4,  1/7,  1,    1/2],
        [1/3,  1/5,  2,    1  ],
    ])

    print('===== 层次分析法（AHP）：选择旅游地·准则层赋权 =====')
    print('判断矩阵 A =')
    print(np.round(A, 4))

    res = ahp(A)

    print('\n----- 权重结果 -----')
    for name, wj in zip(names, res['weights']):
        print(f'{name}：权重 = {wj:.4f}')

    print('\n----- 一致性检验 -----')
    print(f"最大特征值 λmax = {res['lambda_max']:.4f}")
    print(f"一致性指标 CI   = {res['CI']:.4f}")
    print(f"随机一致性指标 RI = {RI_DICT[len(A)]:.2f}")
    print(f"一致性比率 CR   = {res['CR']:.4f}")
    if res['consistent']:
        print('结论：CR < 0.1，判断矩阵通过一致性检验，权重可用 [V]')
    else:
        print('结论：CR >= 0.1，判断矩阵未通过一致性检验，请调整标度后重算 [X]')

    # ----- 可视化：准则权重柱状图 -----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, res['weights'], color='#4C72B0', edgecolor='black')
    for i, wj in enumerate(res['weights']):
        ax.text(i, wj + 0.005, f'{wj:.4f}', ha='center', fontsize=10)
    ax.set_title('AHP 准则层权重（选择旅游地）')
    ax.set_xlabel('准则')
    ax.set_ylabel('权重')
    ax.set_ylim(0, max(res['weights']) * 1.2)
    plt.tight_layout()
    fig.savefig('ahp_demo.png', dpi=150, bbox_inches='tight')
    print('ahp_demo.png 已保存')
    plt.close(fig)
    print('\nahp demo 运行完毕')
