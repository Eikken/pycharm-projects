#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   空间立方体距离计算测试.py    
@Time    :   2021/11/4 17:12  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

from scipy.spatial import distance
import numpy as np


# pList = np.array([
#     [1, 1, 0],
#     [1, 1, 1],
#     [0, 1, 0],
#     [1, 0, 0]
#     ])
#
# print(distance.cdist(np.array([[0,0,0]]), pList))
#
# OA = 1.42
# OB = None
# AB = 0.71
# theta = None
# print('∠AOB degree>>', np.rad2deg(np.arctan(AB/OA)), np.arctan(AB/OA))
#

dic = {"A": [True, False, True, False, True, False]}
print(dic['A'])
dic.update({'A': [x + 0 for x in dic['A']]})
print(dic.get('A'))