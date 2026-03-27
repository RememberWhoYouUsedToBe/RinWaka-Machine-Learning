"""
梯度下降练习_6
我发誓这绝对是我最后一个梯度下降练习了
2026/3/12
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearDataset:
    """
    用于生成数据集的类
    """
    def __init__(self,RandomSeed = 31 , N = 100):
        """
        N: 生成多少个点,默认值100
        RandomSeed: 随机种子,默认值42
        """
    
        # 设置随机种子以便结果可复现
        np.random.seed(RandomSeed)

        # 生成特征 X：N 个点，范围在 0 到 10 之间
        self.X = np.linspace(0, 10, N ).reshape(-1, 1)

        # 真实的参数：w = 2, b = 1
        self.true_w = 2.0
        self.true_b = 1.0



    # 生成标签 y = 2x + 1 + 高斯噪声
    def generate_y (self, true_w, true_b, X, user_input = False, N = 100):
        """
        控制是否启用噪声
        """
        if user_input == True:
            y = true_w * X.squeeze() + true_b + np.random.randn(N) * 2  # 噪声标准差为2
        else:
            y = true_w * X.squeeze() + true_b   # 无噪声
        
        return y

#---

class GradientDescentLoop:
    """
    完整的正常循环
    """

    def __init__(self, n_features):
        """
        参数初始化

        n_features: 去除首项(偏置项)后特征数
        """

        self.k = [0.0] * (n_features + 1)   # 列表推导式, 例如 n_features=2，则 k = [0.0, 0.0, 0.0]


    def compute_predictions(self, X):
        """
        进行预测以免在其他模块重复写预测, 返回一个列表/数组

        X: 二维列表或二维数组，形状 (m, n_features)
        """
        
        m = len(X)
        predictions = []
        for i in range(m):
            pred = self.k[0]
            for j in range(1, len(self.k)):  # j从1开始，对应特征权重
                pred += self.k[j] * X[i][j-1]   # 注意 X[i][j-1] 对应第 j-1 个特征
            predictions.append(pred)
        return predictions


    def compute_loss(self, X, y):
        """
        损失计算
        """

        m = len(X)
        predictions = self.compute_predictions(X)
        total_error = 0.0
        for i in range(m):
            error = predictions[i] - y[i]
            total_error += error ** 2
        loss = total_error / (2 * m)
        return loss
    
    def compute_gradients(self, X, y):
        m = len(X)
        predictions = self.compute_predictions(X)
        gradients = [0.0] * len(self.k)      # 初始化，长度 = 特征数 + 1

        for i in range(m):
            error = predictions[i] - y[i]
            # 偏置的梯度（对应 self.k[0]）
            gradients[0] += error
            # 权重的梯度（对应 self.k[1], self.k[2], ...）
            for j in range(1, len(self.k)):
                gradients[j] += error * X[i][j-1]   # X[i][j-1] 对应第 j-1 个特征

        # 取平均值
        for j in range(len(self.k)):
            gradients[j] /= m

        return gradients
    

    def update_parameters(self, X, y, learning_rate=0.01):
        """
        参数更新
        """

        gradients = self.compute_gradients(X, y)

        for j in range(len(self.k)):
            self.k[j] -= learning_rate * gradients[j]

    def fit(self, X, y, learning_rate=0.01, epochs=1000, verbose=True):
        """
        完整的训练循环
        """

        loss_history = []
        for epoch in range(epochs):
            self.update_parameters(X, y, learning_rate)
            loss = self.compute_loss(X, y)
            loss_history.append(loss)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.6f}")
        return loss_history
    
#---

def main():
    # 1. 创建数据集实例
    dataset = LinearDataset(N=100, RandomSeed=42)

    # 2. 获取特征 X（已经是二维数组）
    X = dataset.X

    # 3. 生成标签 y（调用 generate_y 方法，返回数组）
    y = dataset.generate_y(
        true_w=dataset.true_w,   # 传入真实权重
        true_b=dataset.true_b,   # 传入真实偏置
        X=dataset.X,             # 传入特征
        user_input=False,         # 启用噪声
        N=1000                    # 样本数量
    )

    # 4. 特征数量
    n_features = X.shape[1]       # 应为 1

    # 5. 创建模型并训练
    model = GradientDescentLoop(n_features)
    print("初始参数:", model.k)
    print("初始损失:", model.compute_loss(X, y))

    loss_history = model.fit(X, y, learning_rate=0.035, epochs=2000)

    print("\n训练完成！")
    print("最终参数:", model.k)
    print("真实关系应为: y = 1 + 2*x")
    print("最终损失:", loss_history[-1])

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == "__main__":
    main()