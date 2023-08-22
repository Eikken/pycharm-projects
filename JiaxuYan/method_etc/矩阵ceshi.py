#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   矩阵ceshi.py    
@Time    :   2021/8/3 14:34  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np


theta = np.deg2rad(60)

Matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
M = np.array([
    [1, 3],
    [5, 4]
])
print(np.dot(np.dot(M, np.linalg.inv(Matrix)), Matrix))
print(np.dot(np.dot(M, Matrix.T), Matrix))