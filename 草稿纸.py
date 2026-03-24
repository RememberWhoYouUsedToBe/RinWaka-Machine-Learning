from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.describe())

# 箱线图：直观看出中位数、四分位数和异常值
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :4])
plt.xticks(rotation=45)
plt.title("Boxplot of Iris Features")
plt.show()

# 直方图：查看分布形状
df.iloc[:, :4].hist(bins=20, figsize=(10, 8))
plt.suptitle("Histograms of Iris Features")
plt.show()