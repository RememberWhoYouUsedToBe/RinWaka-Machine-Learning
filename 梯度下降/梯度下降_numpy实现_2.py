"""
第三个练习
2026/5/15
"""

import numpy as np

class data : 
    """
    数据集类 —— 无噪声
    真实模型: y = 2*x1 - 3*x2 + 1*x3 + 5
    """

    def __init__(self):
        # 原始列表（10个样本，每个样本3个特征）
        X_list = [
            [1.0, 2.0, 3.0, 1.0],
            [2.0, 1.0, 4.0, 1.0],
            [3.0, 3.0, 1.0, 1.0],
            [4.0, 2.0, 2.0, 1.0],
            [5.0, 1.0, 5.0, 1.0],
            [6.0, 4.0, 2.0, 1.0],
            [7.0, 3.0, 3.0, 1.0],
            [8.0, 5.0, 1.0, 1.0],
            [9.0, 2.0, 4.0, 1.0],
            [10.0, 3.0, 5.0, 1.0]
        ]
        # 对应的目标值 y = 2*x1 - 3*x2 + 1*x3 + 5
        y_list = [
            2*1.0 - 3*2.0 + 1*3.0 + 5,   # = 2 -6 +3 +5 = 4
            2*2.0 - 3*1.0 + 1*4.0 + 5,   # = 4 -3 +4 +5 = 10
            2*3.0 - 3*3.0 + 1*1.0 + 5,   # = 6 -9 +1 +5 = 3
            2*4.0 - 3*2.0 + 1*2.0 + 5,   # = 8 -6 +2 +5 = 9
            2*5.0 - 3*1.0 + 1*5.0 + 5,   # = 10 -3 +5 +5 = 17
            2*6.0 - 3*4.0 + 1*2.0 + 5,   # = 12 -12 +2 +5 = 7
            2*7.0 - 3*3.0 + 1*3.0 + 5,   # = 14 -9 +3 +5 = 13
            2*8.0 - 3*5.0 + 1*1.0 + 5,   # = 16 -15 +1 +5 = 7
            2*9.0 - 3*2.0 + 1*4.0 + 5,   # = 18 -6 +4 +5 = 21
            2*10.0 - 3*3.0 + 1*5.0 + 5    # = 20 -9 +5 +5 = 21
        ]
        # 转为 NumPy 数组并保存为实例属性
        self.X = np.asarray(X_list)   # 形状 (10, 3)
        self.y = np.asarray(y_list)   # 形状 (10,)

class LinearRegression : 
    
    def __init__(self, n_features = 4, seed = 42) -> None:
        """
        参数初始化
          - Theta :随机K值
        """
        # 设置随机种子
        np.random.seed(seed)

        self.Theta = np.random.randn(n_features)

    #----

    def compute_prediction (self, X):
        """
        预测函数
        """
        predictions = X @ self.Theta    # 预测值

        return predictions
    
    def compute_loss (self, X, y) :
        """
        损失函数
        """

        m = len(X)

        prediction = self.compute_prediction(X)

        Loss = (1 / (2 * m)) * np.sum((prediction - y) ** 2)

        return Loss

    def compute_gradient (self, X, y, Theta) :
        """
        梯度计算函数
        """

        m = len(X)

        error = X @ Theta - y

        gradients = (1 / m) * (X.T @ error)

        return gradients
    
    def update_parameters (self ) :
        """
        参数更新函数
        """

    #----

def main():
    Model = LinearRegression()

    Data = data()
    X = Data.X
    y = Data.y
    Theta = Model.Theta

    grad = Model.compute_gradient(X, y, Theta)

    print(grad)

if __name__ == "__main__":
    main()
