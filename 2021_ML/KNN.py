#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   KNN.py    
@Time    :   2021/2/16 20:24  
@Tips    :   K临近聚类
'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

filmData = pd.read_excel(r'data\电影分类数据.xlsx')

cols = np.array(filmData.columns)
sampleList = cols[-3:]#.tolist()
# print(sampleList)
cloValue = cols[2:5]
trainData = filmData[cloValue]
train = filmData[cols[2:6]]
# 计算唐人街与每个电影的距离
sort_index = np.argsort(np.sqrt(np.sum((trainData-sampleList)**2,axis=1)))[:5] # 选取距离最小的五个样本，看他们的电影类型众数是多少
# print(sort_index)
filmType = cols[5:6].tolist()
value = train[filmType[0]][sort_index].mode().values#[sort_index].mode().values

print(value)
# digits = datasets.load_digits()
#
# clf = KNeighborsClassifier(n_neighbors=6)
#
# x, y = digits.data[:-1], digits.target[:-1]
# clf.fit(x, y)
#
# y_pred = clf.predict([digits.data[-1]]) # 传入一个二维数组
# y_true = digits.target[-1]
#
# print(x)
# print('pred:',y_pred)
# print('true:',y_true)