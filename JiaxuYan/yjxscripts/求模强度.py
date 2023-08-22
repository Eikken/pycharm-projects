#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   求模强度.py    
@Time    :   2023/3/20 15:07  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlwt


def func_here(*args, **kwargs):
    # val_i = []

    pass


if __name__ == '__main__':
    # start here
    data = pd.read_csv('data/model-2380.txt').values

    len_val = len(data)
    ls = []
    for i in data:
        len_i = len(i[0].split())
        val_i = list(pd.to_numeric(i[0].split()))

        col = val_i[3] ** 2 + val_i[4] ** 2 + val_i[5] ** 2

        val_i.append(col)
        ls.append(val_i)

    dataSet = np.array(ls)
    pd.DataFrame(dataSet).to_csv("data/model-2380-text.csv", index=False)
    print('finish')

    # print(dataSet)
    # x =
    # y =
    # z =
    # print(" ",x**2+y**2+z**2)
    #
