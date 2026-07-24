# -*- coding: utf-8 -*-
"""
模块：decision_tree_rf
功能：决策树与随机森林分类，以鸢尾花数据集为例，含混淆矩阵与特征重要性分析
适用题型：通用高频（分类预测、特征筛选、可解释性建模）
依赖：numpy, scikit-learn, matplotlib
用法：直接运行 `python decision_tree_rf.py` 查看 demo；或 import 后调用 train_tree_rf
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def train_tree_rf(X, y, test_size=0.3, seed=None):
    """
    同一训练/测试划分上分别训练决策树与随机森林，并评估对比。

    参数：
        X         : 特征矩阵，shape (n_samples, n_features)
        y         : 标签向量，shape (n_samples,)
        test_size : 测试集比例
        seed      : 随机种子，传入后固定训练/测试划分与模型初始化

    返回：
        dict，键为：
            'tree'        : 训练好的决策树模型
            'rf'          : 训练好的随机森林模型
            'tree_acc'    : 决策树测试集准确率
            'rf_acc'      : 随机森林测试集准确率
            'tree_cm'     : 决策树混淆矩阵
            'rf_cm'       : 随机森林混淆矩阵
            'importances' : 随机森林特征重要性，shape (n_features,)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    # 决策树：限制深度防止过拟合
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)

    # 随机森林：100 棵树
    rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                random_state=seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    return {
        'tree': tree,
        'rf': rf,
        'tree_acc': float(accuracy_score(y_test, tree_pred)),
        'rf_acc': float(accuracy_score(y_test, rf_pred)),
        'tree_cm': confusion_matrix(y_test, tree_pred),
        'rf_cm': confusion_matrix(y_test, rf_pred),
        'importances': rf.feature_importances_,
    }


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现

    print("=" * 60)
    print("决策树 vs 随机森林 demo：鸢尾花三分类")
    print("=" * 60)
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names_cn = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    class_names_cn = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    print(f"数据集：{X.shape[0]} 个样本，{X.shape[1]} 个特征，3 个类别")

    res = train_tree_rf(X, y, test_size=0.3, seed=42)

    print("-" * 60)
    print(f"决策树测试集准确率：{res['tree_acc']:.4f}")
    print(f"决策树混淆矩阵：\n{res['tree_cm']}")
    print("-" * 60)
    print(f"随机森林测试集准确率：{res['rf_acc']:.4f}")
    print(f"随机森林混淆矩阵：\n{res['rf_cm']}")
    print("-" * 60)
    print("随机森林特征重要性：")
    for name, imp in sorted(zip(feature_names_cn, res['importances']),
                            key=lambda t: -t[1]):
        print(f"  {name}：{imp:.4f}")

    # 可视化：混淆矩阵对比 + 特征重要性柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, cm, title in zip(axes[:2],
                             [res['tree_cm'], res['rf_cm']],
                             ['决策树混淆矩阵', '随机森林混淆矩阵']):
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_xticks(range(3), class_names_cn, rotation=20)
        ax.set_yticks(range(3), class_names_cn)
        # 在格子中标注数值
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='black', fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8)

    order = np.argsort(res['importances'])
    axes[2].barh(np.array(feature_names_cn)[order], res['importances'][order],
                 color='steelblue')
    axes[2].set_xlabel('重要性得分')
    axes[2].set_title('随机森林特征重要性')
    for i, v in enumerate(res['importances'][order]):
        axes[2].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('tree_rf_demo.png', dpi=150)
    print("已保存混淆矩阵与特征重要性图：tree_rf_demo.png")
    plt.close(fig)
