#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数组合并.py    
@Time    :   2021/3/29 20:10  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

x = np.array([1, 1, 1, 1])
y = np.array([1, 2, 3, 4])
yy = np.array([3, 4, 8, 2, 7, 9])
xx = np.array([1, 4, 7, 2, 9, 6])
xy = np.stack((y, x), axis=-1)
yx = np.stack((y, y), axis=-1)
print(distance.cdist(xy, yx))#
print(distance.cdist(xy, yx).min(axis=1))
for i,j in zip(xy,yx):
    print(i,'  ',j)

plt.scatter(xy[:, 0], xy[:, 1], 10, marker='*', color='green')
plt.scatter(yx[:, 0], yx[:, 1], 10, marker='.', color='red')
plt.show()
