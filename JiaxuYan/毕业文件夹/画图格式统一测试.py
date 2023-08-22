#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   画图格式统一测试.py    
@Time    :   2023/4/18 17:24  
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


def func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    # start here
    dataSet = np.array(pd.read_excel('data/100Expansion.xls')[['angle', 'over_lap_ratio']].values)
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(dataSet[:, 0], dataSet[:, 1], linewidth=0.8)
    plt.tick_params(labelsize=13)
    plt.xlim(-1, 61)
    plt.ylim(-5, 105)
    plt.show()



