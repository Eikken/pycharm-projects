#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   selectAreaPoints.py    
@Time    :   2021/3/26 19:26  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   use circle get all points in this area
'''

import math
import numpy as np
from matplotlib import pyplot as plt
import copy


def f(x, r):
    return (r ** 2 - x ** 2) ** 0.5

def draw(xList, yList, x1, r):
    plt.figure(figsize=(9, 8), edgecolor='black')
    plt.subplot(111)
    plt.scatter(xList, yList, 1, marker='.', color='green')
    plt.scatter([0.0], [0.0], 2, marker='*', color='red')
    # plt.plot([x1,x1], [f(x1, r),-f(x1, r)], lw=1, color='red')
    plt.plot(x1, f(x1, r), lw=1, color='red')
    plt.plot(x1, -f(x1, r), lw=1, color='red')
    plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='grey')
    plt.axvline(x=0, ymin=0, ymax=1, linestyle='--', color='grey')
    plt.savefig('png/circle_abs_green.png', dpi=600)
    plt.show()
    print('finish')


def mo(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


a = (2.522, 0, 0)
b = (2.522 / 2, 2.522 / 2 * math.sqrt(3), 0)
c = (0, 0, 20)
# 扩胞矩阵
super_x = 50
super_y = 50
super_z = 1

transformtion = np.array([[super_x, 0, 0],
                          [0, super_y, 0],
                          [0, 0, super_z]])

lattice = np.array([a, b, c])
newLattice = np.dot(lattice, transformtion)
# print(newLattice*100)  # 晶胞扩大

C1 = [0, 0, 0.5]
C2 = [1 / float(3), 1 / float(3), 0.5]
Frac1 = 0
Frac2 = 1 / float(3)
allAtoms = []
for i in range(super_x):
    for j in range(super_y):
        newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.5]
        newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.5]
        allAtoms.append(newC1)
        allAtoms.append(newC2)
newAllAtoms = np.dot(np.array(allAtoms), newLattice)
xList = np.array(newAllAtoms).T[0] * 100
yList = np.array(newAllAtoms).T[1] * 100
zList = np.array(newAllAtoms).T[2]
x_mean = np.mean(xList)
y_mean = np.mean(yList)
xList = xList - x_mean
yList = yList - y_mean

# 新加代码功能：绘制最大范围圆，以yMean为半径，切掉xList中xList.abs>r的值
x1 = np.linspace(-y_mean, y_mean, int(y_mean))
r = y_mean
# xDrop = np.delete(xList, np.where(xList.__abs__() > r))  # copy.deepcopy(xList)
# yDrop = np.delete(yList, np.where(xList.__abs__() > r))
# 两步法去除circle以外的点，减少计算复杂度
x_Drop = np.delete(xList, np.where(xList.__abs__() > r)) #
y_Drop = np.delete(yList, np.where(xList.__abs__() > r))
mox = np.delete(x_Drop,np.where(mo(x_Drop, y_Drop) > r))
moy = np.delete(y_Drop,np.where(mo(x_Drop, y_Drop) > r))
# print(xDrop)
# print(yDrop)
draw(mox, moy, x1, r)

