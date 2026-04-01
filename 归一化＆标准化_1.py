from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class IrisDataset:
    def __init__(self, test_size=0.2, random_state=42, print_info=False):
        """
        test_size: 测试集比例
        random_state: 随机种子
        print_info: 是否打印数据基本信息
        """
        # 加载数据
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y   # 分类问题建议分层抽样
            
        )

        if print_info:
            self._print_info()

    def _print_info(self):
        """内部方法：打印数据信息"""
        print("特征形状:", self.X.shape)
        print("标签形状:", self.y.shape)
        print("特征名称:", self.feature_names)
        print("类别名称:", self.target_names)
        print("训练集大小:", self.X_train.shape[0])
        print("测试集大小:", self.X_test.shape[0])
        print("训练集:", self.X_train)
        print("测试集:", self.X_test)
    
    def get_train_data(self):
        """返回训练数据"""
        return self.X_train, self.y_train
    

#========
class Z_score:
    """
    标准化
    """
    def Mu(self, X):
        """
        计算均值
        """
        m = len(X)
        if m == 0:
            return 0
        total = sum(X)   # 更简洁的求和方式
        return total / m

    def Sigma(self, X, mu):
        """
        计算总体标准差
        """
        m = len(X)
        if m == 0:
            return 0
        variance = sum((x - mu) ** 2 for x in X) / m
        return variance ** 0.5   # 开方

    def Zscore(self, X):
        """
        标准化计算
        """
        mu = self.Mu(X)
        sigma = self.Sigma(X, mu)
        if sigma == 0:
            # 所有值相同，标准化后全为0
            return [0.0] * len(X)
        return [(x - mu) / sigma for x in X]

#========
class Normalization:
    """
    归一化 (Min-Max scaling)
    """
    def Max(self, X):
        """返回最大值"""
        return max(X)

    def Min(self, X):
        """返回最小值"""
        return min(X)

    def Normal(self, X):
        """
        对列表 X 进行 Min-Max 归一化，返回归一化后的列表
        公式: (x - min) / (max - min)
        """
        if not X:                     # 处理空列表
            return []

        max_val = self.Max(X)
        min_val = self.Min(X)

        if max_val == min_val:        # 所有值相同，归一化后全为0
            return [0.0] * len(X)

        # 列表推导式计算归一化结果
        return [(x - min_val) / (max_val - min_val) for x in X]

#========

def main():

    # 加载数据
    iris_data = IrisDataset(print_info=True)
    X_train, y_train = iris_data.get_train_data()
    
    # 选择第一个特征列（花萼长度）进行测试
    feature_index = 0
    feature_data = X_train[:, feature_index].tolist()  # 转换为列表
    
    # 实例化 Z_score 并标准化
    zs = Z_score()
    standardized = zs.Zscore(feature_data)
    
    # 打印部分结果
    print("\n原始特征值（前10个）:", feature_data[:10])
    print("标准化后（前10个）:", standardized[:10])
    
    # 验证均值和标准差是否接近 0 和 1
    import numpy as np
    mean = np.mean(standardized)
    std = np.std(standardized)
    print(f"标准化后均值: {mean:.6f}（应接近 0）")
    print(f"标准化后标准差: {std:.6f}（应接近 1）")

    # 测试归一化
    norm = Normalization()
    normalized = norm.Normal(feature_data)   # 使用相同特征数据

    print("\n归一化后（前10个）:", normalized[:10])
    print(f"归一化后最小值: {min(normalized):.4f}（应为 0）")
    print(f"归一化后最大值: {max(normalized):.4f}（应为 1）")

if __name__ == "__main__":
    main()