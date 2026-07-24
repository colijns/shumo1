"""
模块：plot_style
功能：统一 Matplotlib 中文学术绘图风格（中文字体、负号、白底网格），并提供图片保存工具
适用题型：通用（所有需要出图的题目）
依赖：matplotlib, numpy（仅 demo 用）
用法：直接运行 `python plot_style.py` 查看 demo；或 import 后调用 set_chinese_style() 与 save_fig()
"""

import os

import matplotlib
import matplotlib.pyplot as plt


def set_chinese_style():
    """
    设置中文学术绘图风格（直接修改 plt.rcParams）。

    完成四件事：
        0. 若当前后端非 Agg，设置 Agg 后端（确保无头环境下可保存图片）；
        1. 中文字体按 SimHei -> Microsoft YaHei -> Noto Sans CJK SC 顺序尝试，
           matplotlib 会自动选用列表中第一个系统已安装的字体；
        2. 关闭 unicode_minus，使坐标轴负号 '-' 正常显示（否则变方块）；
        3. 若 seaborn-v0_8-whitegrid 风格可用则启用（学术论文常用白底网格风），
           旧版本 matplotlib 自动回退为 seaborn-whitegrid 或默认风格。

    参数：无
    返回：None
    """
    # 0. 若后端非 Agg，切换为 Agg（仅在首次调用时生效，不触发 rcParams 重置）
    if matplotlib.get_backend().lower() != 'agg':
        matplotlib.use('Agg')
    # 1. 候选中文字体，按优先级排列
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
    # 2. 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    # 3. 尝试启用学术白底网格风格，找不到时逐级回退
    for style in ['seaborn-v0_8-whitegrid', 'seaborn-whitegrid']:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue  # 该风格不存在，尝试下一个


def save_fig(fig, path, dpi=300):
    """
    保存图片到指定路径并打印中文提示。

    参数：
        fig  : matplotlib.figure.Figure，待保存的图对象
        path : str，保存路径（相对/绝对均可，如 'result.png'）
        dpi  : int，分辨率，默认 300（竞赛论文常用高清图）
    返回：None
    """
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'图片已保存到：{os.path.abspath(path)}（dpi={dpi}）')


if __name__ == '__main__':
    import numpy as np

    np.random.seed(42)  # 固定随机种子，保证结果可复现
    set_chinese_style()

    # ===== demo：画一张带中文标题的示例折线图 =====
    x = np.arange(1, 13)                                       # 月份 1~12
    y = 20 + 10 * np.sin(x / 2.0) + np.random.randn(12) * 1.5  # 模拟月销售额

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, 'o-', color='#1f77b4', linewidth=1.8, label='月销售额（万元）')
    ax.axhline(y.mean(), color='gray', linestyle='--', linewidth=1, label='平均值')
    ax.set_title('中文学术风格示例折线图')
    ax.set_xlabel('月份')
    ax.set_ylabel('销售额（万元）')
    ax.legend(loc='best')

    # 保存到当前工作目录
    save_fig(fig, 'plot_style_demo.png', dpi=300)
    plt.close(fig)
    print('plot_style demo 运行完毕')
