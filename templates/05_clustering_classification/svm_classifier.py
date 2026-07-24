# -*- coding: utf-8 -*-
"""
模块：svm_classifier
功能：支持向量机（SVM）分类，含标准化管线与 C/gamma 网格搜索（交叉验证选参）
适用题型：通用高频（小样本分类、高维分类、需要核技巧的非线性分类）
依赖：numpy, scikit-learn, matplotlib
用法：直接运行 `python svm_classifier.py` 查看 demo；或 import 后调用 svm_grid_search
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt

# 中文字体与负号设置（SPEC 约定 5，单文件独立设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def svm_grid_search(X, y, C_grid=(0.1, 1, 10, 100), gamma_grid=(0.01, 0.1, 1, 10),
                    test_size=0.3, cv=5, random_state=42):
    """
    SVM（RBF 核）分类：标准化 + 网格搜索最优 C 与 gamma，返回评估结果。

    参数：
        X            : 特征矩阵，shape (n_samples, n_features)
        y            : 标签向量，shape (n_samples,)
        C_grid       : 候选惩罚系数 C 网格
        gamma_grid   : 候选 RBF 核参数 gamma 网格
        test_size    : 测试集比例
        cv           : 交叉验证折数
        random_state : 随机种子

    返回：
        dict，键为：
            'best_params' : 最优参数字典 {'C':..., 'gamma':...}
            'best_score'  : 交叉验证最优平均准确率
            'test_acc'    : 最优模型在测试集上的准确率
            'cm'          : 测试集混淆矩阵
            'model'       : 训练好的最优管线模型（含标准化）
            'cv_results'  : (len(C_grid), len(gamma_grid)) 的交叉验证平均准确率矩阵
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 管线：先标准化（SVM 对量纲敏感），再 SVC
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', random_state=random_state)),
    ])
    param_grid = {'svc__C': list(C_grid), 'svc__gamma': list(gamma_grid)}
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='accuracy', n_jobs=None)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    # 整理网格上各 (C, gamma) 组合的交叉验证平均准确率，便于画热力图
    mean_scores = gs.cv_results_['mean_test_score'].reshape(
        len(C_grid), len(gamma_grid))

    return {
        'best_params': {'C': gs.best_params_['svc__C'],
                        'gamma': gs.best_params_['svc__gamma']},
        'best_score': float(gs.best_score_),
        'test_acc': float(accuracy_score(y_test, y_pred)),
        'cm': confusion_matrix(y_test, y_pred),
        'model': best,
        'cv_results': mean_scores,
    }


if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，保证可复现

    print("=" * 60)
    print("SVM demo：鸢尾花分类 + C/gamma 网格搜索")
    print("=" * 60)
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names_cn = ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾']
    print(f"数据集：{X.shape[0]} 个样本，{X.shape[1]} 个特征，3 个类别")

    # 小网格保证 demo 运行时间 < 30 秒
    C_grid = (0.1, 1, 10, 100)
    gamma_grid = (0.01, 0.1, 1, 10)
    res = svm_grid_search(X, y, C_grid=C_grid, gamma_grid=gamma_grid,
                          test_size=0.3, cv=5, random_state=42)

    print("-" * 60)
    print(f"最优参数：C = {res['best_params']['C']}，gamma = {res['best_params']['gamma']}")
    print(f"交叉验证最优平均准确率：{res['best_score']:.4f}")
    print(f"测试集准确率：{res['test_acc']:.4f}")
    print(f"测试集混淆矩阵：\n{res['cm']}")

    # 可视化：网格搜索准确率热力图 + 混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    im0 = axes[0].imshow(res['cv_results'], cmap='viridis', aspect='auto')
    axes[0].set_xticks(range(len(gamma_grid)), [str(g) for g in gamma_grid])
    axes[0].set_yticks(range(len(C_grid)), [str(c) for c in C_grid])
    axes[0].set_xlabel('gamma')
    axes[0].set_ylabel('C')
    axes[0].set_title('网格搜索交叉验证平均准确率')
    for i in range(len(C_grid)):
        for j in range(len(gamma_grid)):
            axes[0].text(j, i, f"{res['cv_results'][i, j]:.3f}",
                         ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(res['cm'], cmap='Blues')
    axes[1].set_title(f'测试集混淆矩阵（准确率 {res["test_acc"]:.4f}）')
    axes[1].set_xlabel('预测类别')
    axes[1].set_ylabel('真实类别')
    axes[1].set_xticks(range(3), class_names_cn, rotation=20)
    axes[1].set_yticks(range(3), class_names_cn)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(res['cm'][i, j]), ha='center', va='center',
                         color='black', fontsize=12)
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig('svm_demo.png', dpi=150)
    print("已保存网格搜索热力图与混淆矩阵：svm_demo.png")
    plt.close(fig)
