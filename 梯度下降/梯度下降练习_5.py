import numpy as np   # 这里只用于生成数据，不用于向量化计算

# 数据集
class GenerateLinearDataset:
    X_train = np.array([
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10]
    ], dtype=float)

    y_label = np.array([5, 9, 13, 17, 21], dtype=float)

# 纯循环实现的梯度下降类
class GradientDescentLoop:
    def __init__(self, n_features):
        """
        初始化参数
        :param n_features: 特征数量（不包括截距）
        """
        # 参数列表：第一个是截距，后面是每个特征的权重
        self.k = [0.0] * (n_features + 1)   # 例如 n_features=2，则 k = [0.0, 0.0, 0.0]

    def compute_predictions(self, X):
        """
        计算所有样本的预测值（返回列表）
        X: 二维列表或二维数组，形状 (m, n_features)
        """
        m = len(X)
        predictions = []
        for i in range(m):
            # 预测值 = 截距 + 各特征加权和
            pred = self.k[0]  # 先加截距
            for j in range(1, len(self.k)):  # j从1开始，对应特征权重
                pred += self.k[j] * X[i][j-1]   # 注意 X[i][j-1] 对应第 j-1 个特征
            predictions.append(pred)
        return predictions

    def compute_loss(self, X, y):
        """
        计算均方误差损失
        """
        m = len(y)
        predictions = self.compute_predictions(X)
        total_error = 0.0
        for i in range(m):
            error = predictions[i] - y[i]
            total_error += error ** 2
        loss = total_error / (2 * m)
        return loss

    def compute_gradients(self, X, y):
        """
        计算所有参数的梯度（返回列表，顺序与 self.k 一致）
        """
        m = len(X)
        predictions = self.compute_predictions(X)
        # 初始化梯度列表，长度与参数个数相同
        gradients = [0.0] * len(self.k)

        for i in range(m):
            error = predictions[i] - y[i]
            # 对截距的梯度：error * 1
            gradients[0] += error
            # 对每个特征权重的梯度：error * X[i][j]
            for j in range(1, len(self.k)):
                gradients[j] += error * X[i][j-1]   # X[i][j-1] 是第 j-1 个特征

        # 求平均
        for j in range(len(gradients)):
            gradients[j] /= m
        return gradients

    def update_parameters(self, X, y, learning_rate=0.01):
        """
        执行一次参数更新
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


# 主程序
def main():
    X = GenerateLinearDataset.X_train
    y = GenerateLinearDataset.y_label

    # 注意：X 是二维数组，第一维是样本，第二维是特征
    n_features = X.shape[1]   # 得到特征数量

    model = GradientDescentLoop(n_features)
    print("初始参数:", model.k)
    print("初始损失:", model.compute_loss(X, y))

    # 训练
    loss_history = model.fit(X, y, learning_rate=0.025, epochs=2000)

    print("\n训练完成！")
    print("最终参数:", model.k)
    print("真实关系应为: y = 1 + 2*x1 + 1*x2")
    print("最终损失:", loss_history[-1])

if __name__ == "__main__":
    main()