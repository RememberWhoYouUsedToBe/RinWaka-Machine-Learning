"""
梯度下降----numpy库实现
使用numpy库替代python原生实现,以达到优化和加速
本来打算不练梯度下降了......算了再练就练吧
2026/4/2
"""

# === 使用的库 ===

import numpy as np
from numpy.typing import ArrayLike   # 用于灵活接收 list / np.ndarray


# === 数据集 ===

class data:

    """
    ***数据集类***

    真实模型: y = 1.0 + 0.5\*x1 + 1.0\*x2 
    """

    X_train = [
    [1, 1, 2],
    [1, 2, 3],
    [1, 3, 1],
    [1, 4, 4],
    [1, 5, 2],
    [1, 6, 5],
    [1, 7, 3],
    [1, 8, 6],
    [1, 9, 4],
    [1, 10, 5]
    ]
    """
    特征 (每行: 截距1, x1, x2)
    """

    y_labels = [
    2.5, 3.0, 2.0, 4.5, 3.5,
    5.5, 4.0, 6.5, 5.0, 6.0
    ]
    """
    标签
    """


# === 主程序 ===

class LinearRegression:
    """
    ***使用梯度下降的线性回归模型***
    """

    def __init__(self, n_features: int = 3) -> None:
        """
        ## 初始化
            参数初始化

        ### 传参列表:
            - n_features: 特征数量(包含截距项), 默认为3
        
        ### 实参列表: 
            - theta: [theta0, theta1, theta2] 初始化为 0
        """

        self.theta = np.zeros(n_features)
        
    def compute_prediction (self, X: ArrayLike) -> np.ndarray:
        """
        ## 预测函数, 避免多次重复运算预测值

        ### 传参: 
            - X: 训练集特征矩阵, 形状 (m, n) 或可转换为ndarray的列表
        
        ### 返回: 
            - prediction: 返回预测向量
        """
        # 将输入转换为 numpy 数组（若已是数组则不复制）
        X = np.asarray(X)
        prediction = X @ self.theta

        return prediction   # 返回预测向量

    def compute_loss(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        ## 损失函数 
            计算均方误差损失 (MSE), 使用向量化操作

        ### 输入: 
            - X: 训练集特征矩阵，形状 (m, n) 或可转换为ndarray的列表
            - y: 标签集向量，形状 (m,) 或可转换为ndarray的列表

        ### 输出:
            - loss: 损失值 (float)
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        m = len(X_arr)

        predictions = self.compute_prediction(X_arr)   # 传入数组
        loss = (1 / (2 * m)) * np.sum((predictions - y_arr) ** 2)

        return loss
    
    def compute_gradients (self, X, y, k):
        """
        
        """
        
        pass

# === 运行区 ===

def main():
    """
    # 主程序
    """
    # 创建模型实例（特征数为3）
    Model = LinearRegression(n_features=3)
    Data = data()
    X_data = Data.X_train
    y_data = Data.y_labels

    # 计算当前参数下的损失
    loss_value = Model.compute_loss(X_data, y_data)
    print(f"初始损失 (theta = [0, 0, 0]) : {loss_value:.6f}")


if __name__ == "__main__":
    main()