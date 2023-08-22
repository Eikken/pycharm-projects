#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   测试字符串转浮点数.py    
@Time    :   2023/3/20 17:06  
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


def str2flost(*args, **kwargs):
    val_i = []
    row = args[0]
    sprow = row.split()
    for j in range(3):  # xyz分不开用绝对字符位置区分，后面三个split区分
        val_i.append(float(row[(j + 1) * 10 - 6:(j + 1) * 10 + 4]))
    dx, dy, dz = float(sprow[-3]), float(sprow[-2]), float(sprow[-1])
    # col = dx ** 2 + dy ** 2 + dz ** 2
    val_i.append(dx)
    val_i.append(dy)
    val_i.append(dz)
    return val_i


if __name__ == '__main__':
    # start here
    data = pd.read_csv('data/model-2380-first.txt').values

    len_val = len(data)
    ls = []
    ss = '     22.679679-11.008242  2.147714    -0.002149    0.002505    0.014618 '
    # print(ss[4:14], ss[14:24], ss[24:34])
    for i in data[:20]:
        # val_i = i[0].split()
        # print(i[0])
        print(str2flost(i[0]))


        #
        # col = val_i[3] ** 2 + val_i[4] ** 2 + val_i[5] ** 2
        #
        # val_i.append(col)
        # ls.append(val_i)

    # dataSet = np.array(ls)
    # pd.DataFrame(dataSet).to_csv("data/model-2380-text.csv")
    print('finish')