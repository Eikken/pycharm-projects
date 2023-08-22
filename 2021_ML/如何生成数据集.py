#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   如何生成数据集.py    
@Time    :   2021/2/15 21:27  
@Tips    :   sklearn.datasets.make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                    n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
                    flip_y=0.01, class_sep=1.0, hypercube=True,shift=0.0, scale=1.0,
                    shuffle=True, random_state=None)
            n_features :特征个数= n_informative + n_redundant + n_repeated
            n_informative：多信息特征的个数
            n_redundant：冗余信息，informative特征的随机线性组合
            n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
            n_classes：分类类别
            n_clusters_per_class ：某一个类别是由几个cluster构成的
'''

# # ===========生成分类样本数据集===========
from sklearn import datasets
import matplotlib.pyplot as plt # 画图工具
data,target=datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,n_repeated=0, n_classes=2, n_clusters_per_class=1)
print(data.shape)
print(target.shape)
plt.scatter(data[:,0],data[:,1],c=target)
plt.show()

