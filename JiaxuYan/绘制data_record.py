#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制data_record.py    
@Time    :   2021/7/6 12:04
@E-mail  :   iamwxyoung@qq.com
@Tips    :   文件目录：E:\桌面文件备份\twist\data record 0705.xlsx
            一张有用的曲线图 20221130
            归一化公式
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getData():
    return pd.read_excel(r'E:\桌面文件备份\twist\newfolder\data record 0705.xlsx')


def takeSecond(elem):
    return elem[3][1]


def getLm(a_, theta_, m_n=1):
    # a_ = 142 || 246
    theta_ = np.deg2rad(theta_)
    if m_n == 1:
        return a_ / (2 * (np.sin(theta_ / 2.0)))
    else:
        return m_n * a_ / (2 * (np.sin(theta_ / 2.0)))


if __name__ == '__main__':
    # dataSet = getData()['邻位142'].dropna()
    title = ['邻位142', '间位246', '对位284']
    beiwei = ['倍位142', '倍位246']
    # dataSet.columns = ['序号', '邻位142', '间位246', '对位284', '倍位n']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    xRange = np.linspace(1, 30, 290)

    y246 = getLm(246, xRange)
    y492 = getLm(246, xRange, 2)
    y738 = getLm(246, xRange, 3)
    y984 = getLm(246, xRange, 4)

    y142 = getLm(142, xRange)
    y284 = getLm(142, xRange, 2)
    y568 = getLm(142, xRange, 4)
    y710 = getLm(142, xRange, 5)
    y994 = getLm(142, xRange, 7)

    for t in title:
        dataSet = getData()[t].dropna()
        dataList = np.zeros((len(dataSet), 2))
        for i in range(len(dataSet)):
            sp = dataSet[i].split(',')
            dataList[i][0] = sp[0]
            dataList[i][1] = sp[1]
        plt.scatter(dataList[:, 0], dataList[:, 1], marker='.', label=t[2:5])

    for t in beiwei:
        dataSet = getData()[t].dropna()
        null_list = []
        index = 0
        for i in dataSet:
            note = i.split(',')[3].split('*')  # ['142', '5']
            null_list.append(list(map(float, i.split(',')[:3])))
            null_list[index].append(list(map(int, note)))
            index += 1
        null_list.sort(key=takeSecond)
        zeroList = np.zeros((len(null_list), 3))
        for i in range(len(null_list)):
            zeroList[i][:3] = null_list[i][:3]
        dictLength = {}
        for i in null_list:
            if i[2] in dictLength.keys():
                dictLength[i[2]] = np.append(dictLength[i[2]], np.array([[i[0], i[1]]]), axis=0)
            else:
                dictLength[i[2]] = np.array([[i[0], i[1]]])
        for k, v in dictLength.items():
            plt.scatter(v[:, 0], v[:, 1], marker='.', label=int(k))
        # plt.title('倍位246')
    plt.plot(xRange[2:], y142[2:], label='142_')
    plt.plot(xRange[30:], y284[30:], label='284_')
    # plt.plot(xRange[30:], y426[30:], label='426_')
    plt.plot(xRange[130:], y568[130:], label='568_')
    plt.plot(xRange[70:], y710[70:], label='710_')
    plt.plot(xRange[120:], y994[120:], label='994_')

    # plt.title('距离合集')
    # plt.legend(loc=1)
    # plt.show()
    # print('finish')
    # for t in title:
    #     dataSet = getData()[t].dropna()
    #     dataList = np.zeros((len(dataSet), 2))
    #     for i in range(len(dataSet)):
    #         sp = dataSet[i].split(',')
    #         dataList[i][0] = sp[0]
    #         dataList[i][1] = sp[1]
    #     plt.scatter(dataList[:, 0], dataList[:, 1], marker='.', label=t[2:5])
    #
    # for t in beiwei:
    #     dataSet = getData()[t].dropna()
    #     null_list = []
    #     index = 0
    #     for i in dataSet:
    #         note = i.split(',')[3].split('*')  # ['142', '5']
    #         null_list.append(list(map(float, i.split(',')[:3])))
    #         null_list[index].append(list(map(int, note)))
    #         index += 1
    #     null_list.sort(key=takeSecond)
    #     zeroList = np.zeros((len(null_list), 3))
    #     for i in range(len(null_list)):
    #         zeroList[i][:3] = null_list[i][:3]
    #     dictLength = {}
    #     for i in null_list:
    #         if i[2] in dictLength.keys():
    #             dictLength[i[2]] = np.append(dictLength[i[2]], np.array([[i[0], i[1]]]), axis=0)
    #         else:
    #             dictLength[i[2]] = np.array([[i[0], i[1]]])
    #     for k, v in dictLength.items():
    #         plt.scatter(v[:, 0], v[:, 1], marker='.', label=int(k))

    plt.plot(xRange[10:], y246[10:], label='246_')
    plt.plot(xRange[100:], y492[100:], label='492_')
    plt.plot(xRange[140:], y738[140:], label='738_')
    plt.plot(xRange[140:], y984[140:], label='984_')

    plt.title('距离合集')
    plt.legend(loc=1)
    plt.show()