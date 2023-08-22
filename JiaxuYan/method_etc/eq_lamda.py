#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   eq_lamda.py    
@Time    :   2022/11/2 15:39  
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


def get_lamda(a_, theta_):
    theta_ = np.deg2rad(theta_)
    return a_ / (3**0.5 * 2 * (np.sin(theta_ / 2.0)))


if __name__ == '__main__':
    # start here
    a1 = 2.4595
    a2 = 2.46
    x1 = np.linspace(0.5, 1.5, 50)
    # y1 = get_lamda(a, x1)
    print(get_lamda(a1, 1.41))
    print(get_lamda(a2, 1.41))
    # plt.scatter(x1, y1, linewidth=0.8, color='red', label='equation')
    # plt.legend()
    # plt.show()