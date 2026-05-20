"""
第一个np实现完成, 练下一个
啊草泥马的,  好烦啊
2026/4/23
"""

import numpy as np

class data:
    """
    数据集类 —— 无噪声版本
    真实模型: y = 1.0 + 0.5 * x1 + 1.0 * x2
    """

    def __init__(self):
        # 原始列表
        X_list = [
            [1, 1, 2],
            [1, 2, 4],
            [1, 3, 1],
            [1, 4, 3],
            [1, 5, 5],
            [1, 6, 2],
            [1, 7, 6],
            [1, 8, 3],
            [1, 9, 5],
            [1, 10, 4]
        ]
        y_list = [
            1.0 + 0.5*1 + 1.0*2,    # 3.5
            1.0 + 0.5*2 + 1.0*4,    # 6.0
            1.0 + 0.5*3 + 1.0*1,    # 3.5
            1.0 + 0.5*4 + 1.0*3,    # 6.0
            1.0 + 0.5*5 + 1.0*5,    # 8.5
            1.0 + 0.5*6 + 1.0*2,    # 6.0
            1.0 + 0.5*7 + 1.0*6,    # 10.5
            1.0 + 0.5*8 + 1.0*3,    # 8.0
            1.0 + 0.5*9 + 1.0*5,    # 10.5
            1.0 + 0.5*10 + 1.0*4    # 10.0
        ]

        # 转为 NumPy 数组并保存为实例属性
        self.X = np.asarray(X_list)   # 形状 (10, 3)
        self.y = np.asarray(y_list)   # 形状 (10,)


class LinearRegression:

    def __init__(self, n_features: int = 3, seed: int = 45) -> None:
        """
        ## 参数初始化（带随机种子）

        ### 参数:
            - n_features : 特征数量(包含截距项), 默认3
            - seed : 随机种子，用于复现相同初始参数
        """

        # 设置随机种子
        np.random.seed(seed)

        # 用标准正态分布生成参数（也可用均匀分布）
        self.Theta = np.random.randn(n_features)

    def compute_prediction (self, X) :
        """
        预测值计算
        """

        preictions = X @ self.Theta
        
        return preictions
    
    def compute_loss (self, X, y):
        """
        损失计算
        """

        m = len (X)

        predictions = self.compute_prediction(X)
        loss = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        return loss
    
    def compute_gradient (self, X, y, k):
        """
        梯度列表计算
        """

        m = len(X)

        error = X @ k - y

        gradients = (1 / m) * (X.T @ error)

        return gradients
    
    def update_parameters (self, X, y, learning_rate = 0.01):
        """
        参数更新
        """

        grad = self.compute_gradient(X, y, self.Theta)
        self.Theta -= learning_rate * grad


class Train_Case:
    """
    训练实例，封装整个训练流程
    """

    def __init__(self, model, X, y):
        """
        参数:
            model : 已实例化的 LinearRegression 模型
            X     : 特征矩阵 (np.ndarray)
            y     : 标签向量 (np.ndarray)
        """
        self.model = model
        self.X = X
        self.y = y
        self.loss_history = []   # 记录损失变化

    def train(self, epochs=500, learning_rate=0.05, verbose=True):
        for epoch in range(epochs):
            self.model.update_parameters(self.X, self.y, learning_rate)
            loss = self.model.compute_loss(self.X, self.y)
            self.loss_history.append(loss)

            # 提前终止条件
            if loss < 1e-10:
                if verbose:
                    print(f"Epoch {epoch:4d} | Loss: {loss:.10f} → 收敛，停止训练")
                break

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.10f}")

    def evaluate(self):
        """
        输出最终结果
        """
        final_loss = self.model.compute_loss(self.X, self.y)
        print(f"\n=== 训练完成 ===")
        print(f"最终损失: {final_loss:.10f}")
        print(f"学习到的参数: {self.model.Theta}")
        print("真实参数应为: [1.0, 0.5, 1.0]")
        return final_loss

def main():
    Data = data()
    Model = LinearRegression(n_features=3, seed=45)

    print(f"初始随机参数: {Model.Theta}")

    # 创建训练实例
    trainer = Train_Case(Model, Data.X, Data.y)

    # 开始训练
    trainer.train(epochs=10000, learning_rate=0.038575)

    # 输出最终结果
    trainer.evaluate()

    # 可选：查看损失下降曲线
    # import matplotlib.pyplot as plt
    # plt.plot(trainer.loss_history)
    # plt.show()

if __name__ == "__main__":
    main()