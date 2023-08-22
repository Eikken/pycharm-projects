#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   反比例函数拟合.py    
@Time    :   2021/7/6 16:51
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from JiaxuYan.绘制data_record import getData

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    title = ['邻位142', '间位246', '对位284']
    KB_list = [[8100.5325, 4.01543], [8116.86, 3.44], [8124.1552, 2.2172], [8127.89, 1.938]]
    kb2  = [[14031.91586,8.227],[14054.58978,6.05114]]
    KB_array = np.array(KB_list)
    nb_list = [[7.93, 1779.061], [8.61, 1637.95], [10.42, 1354.86], [11.64, 1213.49]]
    #
    # [32527.121216475003, 27921.998399999997, 18012.87690944, 15751.85082]
    # 4605.122816475006,9909.121490559995,2261.0260894400017
    # plt.plot(KB_array[:, 1], KB_array[:, 0], marker='*')
    x = np.linspace(1.3, 30, 300)
    num = 1
    for i in KB_list:
        y = i[0]/x + i[1]
        plt.plot(x, y, label='曲线'+str(num))
        num += 1
        break
    for i in kb2:
        y = i[0] / x + i[1]
        plt.plot(x[7:], y[7:], label='曲线' + str(num))
        num += 1
        break
    for t in title:
        dataSet = getData()[t].dropna()
        dataList = np.zeros((len(dataSet), 2))
        for i in range(len(dataSet)):
            sp = dataSet[i].split(',')
            dataList[i][0] = sp[0]
            dataList[i][1] = sp[1]
        plt.scatter(dataList[:, 0], dataList[:, 1], marker='.', label=t[2:5])
    plt.title('曲线拟合')
    plt.legend()
    plt.show()
    print('finish')



    # # for i in KB_list:
    # #     nb_list.append(i[0] * i[1])
    # print(nb_list)
    # zuo = []
    # you = []
    # for i in nb_list:
    #     zuo.append(i[0] * i[1])
    #     you.append(i[0])
    # res = []
    # print(zuo)
    # print(you)
    # for i in range(len(you)-1):
    #     res.append([you[i + 1] - you[i], zuo[i + 1] - zuo[i]])
    # print(res)
    # for i in range(len(res)):
    #     print('b' + str(i), '=', res[i][1] / res[i][0])
    # print( 14117.6412 - 10.42*6.051)