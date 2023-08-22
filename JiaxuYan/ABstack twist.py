#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   AB stack twist.py
@Time    :   2021/8/7 17:13  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import xlwt
import time
import pandas as pd


def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def overFlowDrop(xL, yL, R):
    xDrop = np.delete(xL, np.where(xL.__abs__() > R))  #
    yDrop = np.delete(yL, np.where(xL.__abs__() > R))
    return xDrop, yDrop


def matrixTransformation(x_, y_, theta):
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    xT, yT = [], []
    for k, v in zip(x_, y_):
        twistMatrix = np.dot([k, v], Matrix)
        # 矩阵 1X2 * 2X2 = 1X2
        xT.append(twistMatrix[0])
        yT.append(twistMatrix[1])
    return np.array(xT), np.array(yT)


def genGraphene(Super=10, bs=1):  # 返回新的坐标的大胞
    a = (2.460, 0, 0)
    b = (2.460 / 2, 2.460 / 2 * math.sqrt(3), 0)
    c = (0, 0, 20)
    # 扩胞矩阵
    super_x = Super
    super_y = Super
    super_z = 1

    extendCellMatrix = np.array([[super_x, 0, 0],
                                 [0, super_y, 0],
                                 [0, 0, super_z]])
    lattice = np.array([a, b, c])
    # 矩阵右乘扩胞矩阵3X3 * 3X3，生成新的大胞
    extendLattice = np.dot(lattice, extendCellMatrix)
    Frac1 = 0
    Frac2 = 1 / float(3)
    allAtoms = []
    for i in range(super_x):
        for j in range(super_y):
            newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.5]
            newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.5]
            allAtoms.append(newC1)
            allAtoms.append(newC2)
    newAllAtoms = np.dot(np.array(allAtoms), extendLattice)
    x_List = np.array(newAllAtoms).T[0] * bs
    y_List = np.array(newAllAtoms).T[1] * bs
    z_List = np.array(newAllAtoms).T[2] * bs
    x_Mean = np.mean(x_List)
    y_Mean = np.mean(y_List)
    x_List = x_List - np.mean(x_List)
    y_List = y_List - np.mean(y_List)
    return x_List, y_List, z_List, x_Mean, y_Mean


def f(x, R):
    return (R ** 2 - x ** 2) ** 0.5


def drawFig(x1, y1, x2, y2, l2x, l2y, Angle, r):
    xIndex = np.linspace(-r, r, int(r))
    fig = plt.figure(figsize=(25, 12), edgecolor='black')
    if Angle == 0.0:
        plt.xticks([])
        plt.yticks([])
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        # ax1.spines['top'].set_visible(False)  # 不显示图表框的上边框
        # ax1.spines['right'].set_visible(False)
        # ax1.spines['left'].set_visible(False)
        # ax1.spines['bottom'].set_visible(False)
        ax1.scatter(x1, y1, marker='.', color='green')
        ax1.scatter(x2, y2, marker='.', color='blue')
        ax1.plot(xIndex, f(xIndex, r), lw=1, color='red')
        ax1.plot(xIndex, -f(xIndex, r), lw=1, color='red')
        ax1.scatter(0, 0, 10, marker='*', color='black')
        plt.show()
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.scatter(x1, y1, marker='.', color='green')
        ax1.scatter(x2, y2, marker='.', color='blue')
        ax1.plot(xIndex, f(xIndex, r), lw=1, color='red')
        ax1.plot(xIndex, -f(xIndex, r), lw=1, color='red')
        ax1.scatter(0, 0, 10, marker='*', color='black')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.scatter(x1, y1, marker='.', color='green')
        ax2.scatter(l2x, l2y, marker='.', color='blue')
        ax2.plot(xIndex, f(xIndex, r), lw=1, color='red')
        ax2.plot(xIndex, -f(xIndex, r), lw=1, color='red')
        ax2.scatter(0, 0, 10, marker='*', color='black')
        plt.savefig('png/对比/%.2f°_%.2f_AA_AB_stack.png' % (Angle, cellLength[str(Angle)]), dpi=1200)
        print('saved %.2f°.png' % Angle)



if __name__ == '__main__':
    t1 = time.time()
    bs = 100
    Super = 40
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    # 绘制圆
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)  # 注意你删除的原子的方式
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    layer2_x = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    layer2_y = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))-142
    # totalArea = calTotal(mox, bs=bs)
    cellLength = {'3.48': 2338.07, '3.89': 2092.21, '4.41': 1846.37, '5.09': 1600.58, '5.36': 2630.40,
                  '6.01': 1354.86, '6.4': 2204.87, '6.84': 2063.08, '7.34': 1109.28, '7.93': 1779.61,
                  '8.61': 1637.95, '9.43': 863.92, '10.42': 1354.86, '11.64': 1213.49, '13.17': 619.09,
                  '15.18': 931.34, '16.43': 994.20, '17.9': 790.78, '21.79': 375.77, '24.43': 1162.55,
                  '26.01': 1262.37, '27.8': 512.09, '29.41': 1398.82}
    # while True:
    #     angle = float(input('输入角度 >> '))
    #     if angle == 0.0:
    #         drawFig(mox, moy, layer2_x, layer2_y, mox,moy, angle, r)
    #         break
    for k in cellLength.keys():
        thetaAngle = np.deg2rad(float(k))
        x_layer2, y_layer2 = matrixTransformation(layer2_x, layer2_y, thetaAngle)
        xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        drawFig(mox,moy, xTwist, yTwist, x_layer2, y_layer2, float(k), r)
    print('finish!')