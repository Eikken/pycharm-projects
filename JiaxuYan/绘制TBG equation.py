#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制TBG equation.py    
@Time    :   2021/7/31 10:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   TBG equation单调性是怎样的？
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import integrate
import math


def getLm(a_, theta_):
    theta_ = np.deg2rad(theta_)
    return a_ / (math.sqrt(3) * 2 * (np.sin(theta_ / 2.0)))


if __name__ == '__main__':
    data = pd.read_excel(r'data/angle_LBM5-15.xls', sheet_name='Sheet2')
    dataSet = np.array(data[['angle', 'lamLen']])
    a = 2.4595
    x1 = np.linspace(0.5, 1.5, 50)
    y1 = getLm(a, x1)
    plt.figure(figsize=(4, 4), dpi=200)
    plt.scatter(dataSet[:, 0], dataSet[:, 1], linewidth=2, linestyle='--', color='green', label='data')
    plt.scatter(x1, y1, linewidth=0.8, color='red', label='equation')
    plt.legend()
    plt.show()
