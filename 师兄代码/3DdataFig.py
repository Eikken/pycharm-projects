#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   3D_Fig.py
@Time    :   2021/3/21 13:10  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   Yu Jia 3D Data.
x =  波长
y = PL
z = laser power
'''
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# z = np.linspace(0, 13, 100)
# x = 5 * np.sin(z)
# y = 5 * np.cos(z)
# zd = 13 * np.random.random(100)
# xd = 5 * np.sin(zd)
# yd = 5 * np.cos(zd)
# fig = plt.figure()
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(x, y, z, 'gray')
# ax1.scatter3D(xd, yd, zd, cmap='Blues')
# plt.show()


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


data = pd.read_csv(r'data/3D.txt')
# print(data.head())
cols = ['x', '3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5']
xList = np.array(data[['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]).reshape(11, 1024)
yList = np.array(data[['3.5', '4.0', '4.5', '5.0', '5.5', '6.0', '6.5', '7.0', '7.5', '8.0', '8.5']]).reshape(11, 1024)
zList = np.array([3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5])
zz = np.expand_dims(zList,axis=1).repeat(1024).reshape(11,1024) # niu B
fig1 = plt.figure()
fig = plt.axes(projection='3d')
# X, Y = np.meshgrid(xList[1], yList[1])
Z = np.expand_dims(zList,axis=1)#.repeat().reshape(1024,1024)
fig.plot_surface(xList, yList, zz, rstride=1, cstride=1, alpha=0.3, cmap='rainbow') #生成表面， alpha 用于控制透明度
# fig.contour(xList[1], yList[1], zList, zdir='z', offset=-6, cmap="rainbow")
# fig.contour(xList[1], yList[1], zList, zdir='y', offset=-6, cmap="rainbow")
# fig.contour(xList[1], yList[1], zList, zdir='x', offset=-6, cmap="rainbow")
# fig = plt.figure()
# ax2 = Axes3D(fig)
# fig.set_xlabel('X')
# fig.set_xlim(500, 1000)  #拉开坐标轴范围显示投影
# fig.set_ylabel('Y')
# fig.set_ylim(0, 70000)
fig.set_zlabel('Z')
fig.set_zlim(3, 10)
plt.savefig('png/3DD.png',dpi=600)
plt.show()

