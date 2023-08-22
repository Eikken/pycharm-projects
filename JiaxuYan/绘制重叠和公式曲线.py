#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制重叠和公式曲线.py    
@Time    :   2021/8/15 12:41  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getData():
    return pd.read_excel(r'E:\桌面文件备份\twist\newfolder\300_0.01.xls')


def getLm(a_, theta_, m_n=1):
    # a_ = 142 || 246
    theta_ = np.deg2rad(theta_)
    if m_n == 1:
        return a_ / (2 * (np.sin(theta_ / 2.0)))
    else:
        return m_n * a_ / (2 * (np.sin(theta_ / 2.0)))


# def testFunc(a_, theta_, m_n=1):
#     cs = 142  # 该参数是为了把曲线归一化
#     # theta_ = np.deg2rad(theta_)
#     if m_n == 1:
#         return a_ * (np.arcsin(2/theta_))
#     else:
#         return a_ * (np.arcsin(2/theta_)) / m_n
#     # return 2 * np.arcsin(a_ / (2 * theta_))


def getLineABC(param, param1):
    k = (param1[218] - param1[0]) / (param[218] - param[0])
    b = param[108] - k * param[0]
    return k, -1, b


def symmetryPoint(A, B, C, x3, y3):
    """计算一般情况的直线对称点，根据斜率关系推导的数学关系式"""
    x = x3 - 2 * A * ((A * x3 + B * y3 + C) / (A * A + B * B))
    y = y3 - 2 * B * ((A * x3 + B * y3 + C) / (A * A + B * B))
    return x, y


#
#
def tFunc142(a_, theta_):
    return a_ * (np.arcsin(2 / theta_)) ** 1.2 - 5


#
# def testFunc246(a_, theta_):
#     return a_ * (np.arcsin(2 / theta_))
#
#
# def testFunc284(a_, theta_, m_n):
#     return a_ * (np.arcsin(2/theta_)) / m_n


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    title = ['angle', 'over_lap_ratio']
    dataSet = np.array(getData()[title])
    angleList = [3.15, 3.48, 3.89, 4.41, 5.09, 6.01, 7.34, 9.43, 13.17, 21.79]
    xRange = np.linspace(0.5, 1.5, 50)
    y142 = tFunc142(142, 30 - xRange)
    # a_ =
    # y246 = testFunc246(246, 30 - xRange[70:])
    # y284 = testFunc284(142, 30 - xRange[55:], 2)
    # A, B, C = getLineABC(xRange, 142 - y142)
    # x1, y1 = symmetryPoint(A, B, C, xRange, 142 - y142)
    # print(xRange[0], 142 - y142[0])
    # plt.scatter(xRange[208], 142 - y142[208])
    # y2 = (-A/B) * xRange - C/B
    y246 = getLm(246, xRange)
    y284 = getLm(142, xRange, 2)
    # y492 = getLm(246, xRange, 2)
    # y568 = getLm(142, xRange, 4)
    # y710 = getLm(142, xRange, 5)
    # plt.plot(x1, y1, label='142.symmetry')
    # for i in xRange:
    #     print(i, np.deg2rad(i))
    plt.plot(xRange, y142 * 0.7, label='142')
    plt.plot(xRange[70:], (y246 - 18) * 0.1, label='246')
    plt.plot(xRange[55:], (y284 - 5) * 0.40, label='284')

    # plt.plot(xRange[:220], y2[:220], label='symmetry-line')
    # plt.plot(xRange, 142 - y142, label='142.T')
    # plt.plot(y142, xRange, label='142.T')
    # plt.plot(xRange[10:], y246[10:], label='246.0')
    # plt.plot(xRange[20:], y284[20:], label='284.0')
    # plt.plot(xRange[100:], y492[100:], label='492.0')
    # plt.plot(xRange[130:], y568[130:], label='568.0')
    # plt.plot(xRange[70:], y710[70:], label='710.0')
    plt.plot(dataSet[100:, 0], dataSet[100:, 1], color='black', linewidth='0.3', label='ratio')  # 筛选出较高的序列
    plt.title('ratio-distance')
    plt.legend(loc='upper right')
    plt.show()

    # print(dataSet[:10, 0], 12/dataSet[:10, 1])
