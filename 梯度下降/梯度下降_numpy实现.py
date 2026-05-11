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

    真实模型: y = 1.0 + 0.5\\*x1 + 1.0\\*x2 
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
            - k: [k0, k1, k2] 初始化为 0
        """

        self.k = np.zeros(n_features)
        
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
        prediction = X @ self.k

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
    
    def compute_gradients (self, X: ArrayLike, y: ArrayLike, k: np.ndarray) -> np.ndarray:
        """
        ## 梯度计算
            计算梯度列表, 用于更新参数

        ### 输入: 
            - X: 训练集特征矩阵，形状 (m, n) 或可转换为ndarray的列表
            - y: 标签集向量，形状 (m,) 或可转换为ndarray的列表
            - k: 待更新的参数
        
        ### 输出: 
            - gradients: 参数列表
        """

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        k_arr = np.asarray(k)

        m = len(X_arr)

        # 预测误差向量 (m,)
        error = X_arr @ k_arr - y_arr

        # 梯度向量 (n,)
        gradients = (1 / m) * (X_arr.T @ error)

        return gradients

    def update_parameters(self, X, y, learning_rate=0.01) -> None:
        """
        ## 完整更新过程
            一次完整的参数更新过程

        ### 输入：
            - X: 训练集特征矩阵，形状 (m, n) 或可转换为ndarray的列表
            - y: 标签集向量，形状 (m,) 或可转换为ndarray的列表
            - k: 每次更新的参数

        使用梯度下降更新模型参数 self.theta
        """
        grad = self.compute_gradients(X, y, self.k)
        self.k -= learning_rate * grad
        
# === 运行区 ===

def main():
    Model = LinearRegression(n_features=3)
    Data = data()
    X_raw = np.asarray(Data.X_train)
    y_raw = np.asarray(Data.y_labels)

    # 提取特征部分（不含截距列）进行标准化
    X_features = X_raw[:, 1:]   # shape (10, 2)

    mean = X_features.mean(axis=0)
    std = X_features.std(axis=0)

    # 标准化特征
    X_features_scaled = (X_features - mean) / std

    # 重新拼接截距列 1
    X_scaled = np.column_stack([np.ones(len(X_raw)), X_features_scaled])

    # 使用标准化后的数据进行训练
    init_loss = Model.compute_loss(X_scaled, y_raw)
    print(f"初始损失 (k = [0,0,0]) : {init_loss:.6f}")

    learning_rate = 0.5   # 标准化后可以用更大的学习率
    epochs = 10000

    for i in range(epochs):
        Model.update_parameters(X_scaled, y_raw, learning_rate)
        if i % 50 == 0:
            loss = Model.compute_loss(X_scaled, y_raw)
            print(f"Epoch {i:4d} | Loss: {loss:.10f}")

    # 将学习到的参数映射回原始尺度
    k_scaled = Model.k
    # 原始模型： y = k0 + k1 * x1 + k2 * x2
    # 标准化后： x1' = (x1 - mean1)/std1, x2' = (x2 - mean2)/std2
    # 缩放后模型： y = k0' + k1' * x1' + k2' * x2'
    # 代回可得原始系数：
    k1 = k_scaled[1] / std[0]
    k2 = k_scaled[2] / std[1]
    k0 = k_scaled[0] - k1 * mean[0] - k2 * mean[1]

    print(f"\n学习到的原始尺度参数：")
    print(f"k0 = {k0:.6f}, k1 = {k1:.6f}, k2 = {k2:.6f}")
    print("真实参数应为：k0 = 1.5, k1 = 0.0, k2 = 0.5")

if __name__ == "__main__":
    main()