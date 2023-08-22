#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   画层次线1.py    
@Time    :   2022/5/29 0:06  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   局部加权回归（Lowess）的大致思路是：以一个点x xx为中心，向前后截取一段长度为f r a c fracfrac的数据，对于该段数据用权值函数w ww做一个加权的线性回归，记( x , y ^ ) (x,\hat{y})(x,
y
^
​
 )为该回归线的中心值，其中y ^ \hat{y}
y
^
​
 为拟合后曲线对应值。对于所有的n nn个数据点则可以做出n nn条加权回归线，每条回归线的中心值的连线则为这段数据的Lowess曲线。
'''

import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def flexibeArea(x_):
    up = x_[:, 0] < 1.59
    down = x_[:, 0] > 1.46
    return up == down


if __name__ == '__main__':
    filePath = r'C:\Users\Celeste\Desktop\data1.xlsx'  # 文件的根目录
    file_data = pd.read_excel(filePath)
    columnList = file_data.columns.values.tolist()
    dataArr = np.array(file_data)
    inDx = flexibeArea(dataArr)
    dataArr = dataArr[inDx]

    canshu = 50
    lowess = sm.nonparametric.lowess
    plt.figure(figsize=(4, 6))
    for i in range(len(columnList) - 1):
        print(i)
        # result = lowess(dataArr[:, i+1], dataArr[:, 0], frac=0.01, it=3, delta=0.0)
        # plt.plot(result[:, 0], result[:, 1] + canshu * i, linewidth=2)
        plt.plot(dataArr[:, 0], dataArr[:, i+1] + canshu * i, linewidth=2)
    plt.yticks([])
    plt.xlim((1.46, 1.59))
    plt.savefig('png/figure0529.png', dpi=300)
    plt.show()

    print('finish')