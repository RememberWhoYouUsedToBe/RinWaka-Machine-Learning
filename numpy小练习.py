"""
numpy小练习
"""

import numpy as np

# 1. 尝试不同形状
a = np.zeros((5,5))
b = np.zeros(10)
c = np.zeros((10,1))   # 10行1列的列向量
print("a.shape:", a.shape)
print("b.shape:", b.shape)
print("c.shape:", c.shape)

# 2. 尝试其他初始化
d = np.ones((3,4))
e = np.eye(4)
f = np.arange(0, 20, 5)   # [0,5,10,15]

# 3. 简单运算
g = np.array([[1,2],[3,4]])
h = np.array([[5,6],[7,8]])
i = g + h
j = (g @ h)   # 矩阵乘法

def PRINT (state = False):
    """
    state: 是否打印
    """
    if state == True:
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)
        print(f)
        print(g)
        print(h)
        print(i)
        print(j)

def V1 (state = False):
    if state == True:
        # 1. 把向量堆成矩阵
        v1 = np.array([1,2,3])
        v2 = np.array([4,5,6])
        print(v1, v2)
        M_from_rows = np.vstack([v1, v2])   # 垂直堆叠 → shape (2,3)
        print(M_from_rows)

        # 2. 矩阵乘法示例（矩阵 × 向量）
        M = np.array([[1,2], [3,4]])
        print(M)
        x = np.array([5,6])      # 一维向量，视为列向量
        print(x)
        y = M @ x                # 结果是一维向量
        print(y,"结果是一维向量, 因为 1*5+2*6=17, 3*5+4*6=39")                 # [17 39]  因为 1*5+2*6=17, 3*5+4*6=39



def main():
    """
    主函数
    """
    PRINT(True)
    V1(True)
    

if __name__ == "__main__":
    main()