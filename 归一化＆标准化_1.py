"""
归一化/标准化练习
2026/3/18
"""

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data          # 特征，形状 (150, 4)
y = iris.target        # 标签，形状 (150,)

# 可选：查看数据
print("特征形状:", X.shape)
print("标签形状:", y.shape)
print("特征名称:", iris.feature_names)
print("类别名称:", iris.target_names)