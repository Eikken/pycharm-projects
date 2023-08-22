# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   totateTest.py    
@Time    :   2021/3/19 16:01  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   矩阵旋转变换x,y坐标的方法
M(θ) = [[cos(θ) -sin(θ)],   =  [[0  -θ],
        [sin(θ) cos(θ)]]        [θ  0]]
也就是说，逆时针旋转30°的新坐标就是：
[x  y] * [[cos(30°) -sin(30°)], = [x    y] * M(30°)
          [sin(30°) cos(30°)]]
'''

import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-10, 10, 100)
y1 = 2 * x
y2 = x ** 2 - 10
# 旋转角度
theta = np.pi*1.0/180.0

Matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
print(Matrix)
xTotateList = []
yTotateList = []
for k, v in zip(x, y1):
    totateMatrix = np.dot([k, v], Matrix)
    # print('\n[',k,',',v,']','*',theta,'=',totateTheta)
    xTotateList.append(totateMatrix[0])
    yTotateList.append(totateMatrix[1])
plt.figure(figsize=(6, 6))
plt.plot(x, y1, lw=0.5, color='red')
# plt.plot(x, y2, lw=0.5, color='blue')
plt.axvline(x=0, color='grey', linestyle='--', lw=0.5)
plt.axhline(y=0, color='grey', linestyle='--', lw=0.5)
plt.plot(xTotateList, yTotateList, lw=0.5, color='green')

plt.show()
