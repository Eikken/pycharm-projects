#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Xiaowei Young
@File    :   np生成行值矩阵.py    
@Time    :   2023/7/21 11:14  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   func(*args, **kwargs)
'''
import numpy as np

if __name__ == '__main__':
    # start here
    # mat = np.zeros((10, 150))
    # for i in range(mat.shape[1]):
    #     mat[:, i] = np.linspace(0, 13, mat.shape[0])
    # mat[i] for i in range(mat.shape[0])
    outer = np.outer(np.linspace(0, 13, 8), [1 for i in range(150)])
