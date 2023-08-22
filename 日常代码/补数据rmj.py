#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   补数据rmj.py    
@Time    :   2023/4/7 9:49  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   

'''

import numpy as np
import phonopy as pp

if __name__ == '__main__':
    # start here
    # 12 个数，3行4列

    a = np.random.uniform() * 0.66
    print(a)
    # mat34 = np.arange(1, 4*5+1).reshape(4, 5)
    # (row, col) = mat34.shape
    # zero_matrix = np.zeros((6, 6))  # 新的0矩阵
    # for x in range(row):
    #     zero_matrix[x, :col] = mat34[x]
    #
    # dit = {}
    # for i in range(int(6/3)):
    #     for j in range(int(6 / 3)):  # 行列遍历
    #         dit['arr%d%d'%(i, j)] = zero_matrix[i*3:(i+1)*3, j*3:(j+1)*3]
    #
    # print(dit)
    # # 中间什么np.digitize()计算
    # print(zero_matrix)
    #
    # new_arr = np.array([])
    # for i in range(2):  # 3行3列的小矩阵
    #     tmp_arr = np.array([])
    #     for j in range(2):  # 行列遍历
    #         if tmp_arr.size == 0:
    #             tmp_arr = dit['arr%d%d' % (i, j)]
    #         else:
    #             tmp_arr = np.hstack((tmp_arr, dit['arr%d%d' % (i, j)]))
    #
    #     if new_arr.size == 0:
    #         new_arr = tmp_arr
    #     else:
    #         new_arr = np.vstack((new_arr, tmp_arr))
    #
    # print(new_arr)