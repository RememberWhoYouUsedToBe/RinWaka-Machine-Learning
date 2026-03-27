from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy

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
    
    def get_train_data(self):
        """返回训练数据"""
        return self.X_train, self.y_train
    

#========
class Z_score :
    """
    标准差处理
    """
    def __init__(self):
         """
         参数初始化
         """
         self.X = None
         

    def Mu (self, X):
        """
        平均值计算
        """
        m = len(X)
        mu = 0.00

        for i in range(m):
            mu += X[i]

        return mu / m

    def Sigma (self, X, Mu):
        """
        总体标准差的标准差计算
        """
        m = len(X)
        for i in range(0, m):
            Sqrt = (X[i] - self.Mu(X)) ** 2
        
        return Sqrt / m

    def Zscore (self):
        """
        
        """

#========
class Normalization :
    """
    """
#========
class Logic :
    """
    """

#========

def main():
    IrisData = IrisDataset(print_info=True)
    ZscoreModle = Z_score()

    print(IrisData)

if __name__ == "__main__":
    main()