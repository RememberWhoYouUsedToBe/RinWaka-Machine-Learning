"""
梯度下降练习5
上个练习太大了练废了,不写了,写得我头大,重写一个小一点的
2026年3月3日18点18分
"""
import numpy as np

class GenerateLinearDataset:
    """
    用于生成数据集的类
    """ 
    X_train = np.array([
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10]
    ], dtype=float)

    y_label = np.array([5, 9, 13, 17, 21], dtype=float)

    #隐藏的真实关系：y = 1 + 2 * x1 + 1 * x2 (x0截距=1，x1系数=2，x2系数=1)

class Gradient_Descent:
    """
    完整的训练
    """

    w = 0.00
    b = 0.00
    k = [0.0 for _ in range(GenerateLinearDataset.X_train)]   #初始化多元线性回归的各个系数