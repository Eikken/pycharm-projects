#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   支持向量机.py    
@Time    :   2021/2/15 17:47  
@Tips    :   支持向量机分类器,核心思想就是找到不同类别之间的分界面，使得两类样本尽量落在面的两边，而且离分界面尽量远。
'''

from sklearn import svm, datasets
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


digits = datasets.load_digits()

# svm函数
clf = svm.SVC(gamma = 0.001, C = 100)

x, y = digits.data[:-1], digits.target[:-1]

g = clf.fit(x, y)

y_pred = clf.predict([digits.data[1]]) # 传入一个二维数组
y_true = digits.target[1]

print(digits.data[1])
print(digits.target[1])
print('y_pred:',y_pred)
print('y_true:',y_true)
print(g)
# print(len(digits.data[:,0]))
# print(len(digits.target[:]))
plt.scatter(digits.data[:,1],digits.target[:],color='red') # 散点数据
plt.title('SVM')
# plt.show()