#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   线性回归.py
@Time    :   2021/2/14 20:52  
@Tips    :   线性回归算法
'''
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets
import numpy as np
digits = datasets.load_digits() # 创建线性回归数据模型

clf = linear_model.LinearRegression()

x, y = digits.data[:-1], digits.target[:-1]
dataSet = np.array([
    [5,2], [4,1.5], [3,1], [2,1], [1,0.5]
])
# xx = [[5,4,3,2,1]]
# yy = [2,1.5,1,1,0.5]
xx, yy = dataSet[:,0].reshape(-1,1),dataSet[:,1]
#注意： dataSet[:, 0]中添加了一个reshape的函数，主要的原因是在之后调用fit函数的时候对特征矩阵x是要求是矩阵的形式。
# print(xx,'\n',yy)
# [[5.]
#  [4.]
#  [3.]
#  [2.]
#  [1.]]
#  [2.  1.5 1.  1.  0.5]
clf.fit(xx, yy)
b_ = clf.coef_[0] # b^ [0.35]
a_ = clf.intercept_ # a^ 0.14999999

print('y= '+str(round(b_,6))+'x + '+str(round(a_,6)))
plt.scatter(xx,yy,color='red') # 散点数据
plt.plot(xx,clf.predict(xx),color='blue') # 拟合的函数
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print('预测 6 的结果为：%3f'%clf.predict([[6]]))
# Y = b^*X + a^
# Y = 0.35*X + 0.1499
# 预测
# y_pred = clf.predict([digits.data[2]])
# y_true = digits.target[2]

# print('pred_y:',y_pred,' ,true_y:',y_true)