#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制data decord2.py    
@Time    :   2022/11/30 16:11  
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


def getData():
    return pd.read_excel(r'E:\桌面文件备份\twist\newfolder\data record 0705.xlsx', sheet_name='Sheet2')


if __name__ == '__main__':
    # start here
    dataSet = getData()['data'].dropna()
    data_arr = np.zeros((dataSet.shape[0], 3))
    for i in range(len(dataSet)):
        for j in range(3):
            data_arr[i][j] = float(dataSet[i].split(',')[j])

    data_sort = data_arr[data_arr[:, 0].argsort()]
    plt.plot(data_sort[:, 0], data_sort[:, 1])
    plt.show()

    plt.figure(figsize=(4, 6))
    x = np.linspace(0, 4, 400)  # 36>>30
    # y = 0.375 * x ** 2 - 3 * x + 37
    y = 0.46*x**2 - 3.68*x + 37.36
    Xb = []
    freq = -1.79 * x + 37.16
    comp = np.stack((x, freq, y), axis=-1)
    plt.plot(comp[:, 0], comp[:, 1])
    plt.plot(comp[:, 0], comp[:, 2])
    data_5 = data_sort[np.where(data_sort[:, 0] < 4)]
    k = (36 - 30) / (np.max(data_5[:, 1]) - np.min(data_5[:, 1]))  # 系数
    norm_5 = k * (data_5[:, 1] - np.min(data_5[:, 1])) + 30
    data_5[:, 1] = norm_5
    plt.scatter(data_5[:, 0], data_5[:, 1])
    plt.show()

    for i in range(len(data_sort)):
        print([data_sort[i], data_sort[i][1]/246] if np.abs(data_sort[i][1]/142-int(data_sort[i][1]/142)) < 0.1 else 0, end=' ')