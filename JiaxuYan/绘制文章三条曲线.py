#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制文章三条曲线.py    
@Time    :   2023/2/10 1:49  
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
    # file1 = pd.read_excel(r'E:\桌面文件备份\twist\newfolder\180_step_0.001-1.xls')
    fitAngle1 = np.array([[21.787, 14.24],
                          [13.174, 5.222],
                          [9.43, 2.731],
                          [7.34, 1.619],
                          [6.01, 1.098],
                          [5.096, 0.878],
                          [4.41, 0.784]])
    fitAngle2 = np.array([[27.8, 7.664],
                          [17.9, 3.183],
                          [15.18, 2.3],
                          [11.64, 1.398],
                          [10.42, 1.088],
                          [8.613, 0.876],
                          [7.93, 0.824]])
    fitAngle3 = np.array([[16.43, 2.055], [10.99, 0.917], [8.26, 0.761]])

    f1 = np.polyfit(fitAngle1[:, 0], fitAngle1[:, 1], 3)
    print("f1 is :\n", f1)
    p1 = np.poly1d(f1)
    print('p1 is :\n', p1)
    xrange1 = np.linspace(4, 22.5, 50)

    f2 = np.polyfit(fitAngle2[:, 0], fitAngle2[:, 1], 3)
    print("f2 is :\n", f2)
    p2 = np.poly1d(f2)
    print('p2 is :\n', p2)
    xrange2 = np.linspace(7, 28.5, 50)

    f3 = np.polyfit(fitAngle3[:, 0], fitAngle3[:, 1], 3)
    print("f3 is :\n", f3)
    p3 = np.poly1d(f3)
    print('p3 is :\n', p3)
    xrange3 = np.linspace(7, 17, 50)

    file2 = pd.read_excel(r'E:\桌面文件备份\twist\newfolder\180_step_0.001-2.xls')
    dataSet = np.array(file2[['angle', 'over_lap_ratio']])
    plt.figure(figsize=(10, 6))
    plt.plot(dataSet[:, 0], dataSet[:, 1])
    plt.plot(fitAngle1[:, 0], fitAngle1[:, 1], 's', label='origin value')
    plt.plot(xrange1, p1(xrange1), 'r', label='fit value')

    plt.plot(fitAngle2[:, 0], fitAngle2[:, 1], 's', label='origin value')
    plt.plot(xrange2, p2(xrange2), label='fit value')

    plt.plot(fitAngle3[:, 0], fitAngle3[:, 1], 's', label='origin value')
    plt.plot(xrange3, p3(xrange3), label='fit value')

    plt.ylim(0, 20)
    plt.xlim(1, 30)
    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    plt.show()
    print('finished')
