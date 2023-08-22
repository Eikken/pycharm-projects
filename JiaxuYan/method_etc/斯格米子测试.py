#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   斯格米子测试.py
@Time    :   2021/5/12 18:58  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
from math import *
import matplotlib.pyplot as plt


def thetaFunc(r, c, w):  # 360°畴壁表达式
    return 2 * (np.arctan(np.cosh(c / (w / 2)) / np.sinh(r / (w / 2))))


def R0(c, w):
    return w / 2 * (np.arccosh(np.sinh(c / (w / 2))))


def RBH(c, w):
    return (w / 2) * (np.arccosh(np.sinh(c / (w / 2))) +
                      2 * np.tanh(c / (w / 2)) * np.arctan(
                np.cosh(c / (w / 2)) / (np.sinh(c / (w / 2)) ** 2 - 1) ** 0.5))


def RBHr0_0(c, w):
    return (np.pi * w / 4) * np.cosh(2 * c / w)


def Sy180x(x, w0):
    return np.arcsin(np.tanh(x/(w0/2)))


def thetaPCW(p, c, w):
    return np.arcsin(np.tanh((-p+c) / (w / 2))) + np.arcsin(np.tanh((-p-c) / (w / 2))) + np.pi
# def func1(x):
#     return np.arcsin (
#
#     )
def main():
    R = 1
    w_ = np.arange(0.01, 4 * R / (pi * (2 ** 0.5)), 0.01)
    x_ = np.arange(0, 360, 0.1)
    # w_ = np.arange(4 * R / (pi * (2 ** 0.5)), 4 * R / pi, 0.001)
    c_ = (w_ / 2) * np.arcsinh(1) + 0.5
    r0 = R0(c_, w_)
    # r_ = np.arange(0, 90, 1)
    r_ = 0
    fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1, projection='polar')  # 指定绘制极坐标图
    # ax1.plot(w_, RBHr0_0(c_, w_), 1, color='blue')
    ax2 = fig.add_subplot(1, 1, 1)  # 指定绘制极坐标图
    ax2.scatter(w_, thetaFunc(r0, c_, w_), 1)
    plt.show()
    print('finish')


if __name__ == '__main__':  # 4*2/(pi*(2**0.5))
    # main()
    # A = 0
    # Keff = 0
    # w0 = 2 * (A/Keff) ** 0.5
    x = np.arange(0.1,2,0.1)
    y = np.arange(0.1,2,0.1)
    p = (x**2 + y **2) **0.5
    c = 2
    w = 1.5
    # ww = np.arange(1,3,0.1)
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    SXyz = -np.sin(thetaPCW(p, c, w)) * x / p
    SxYz = -np.sin(thetaPCW(p, c, w)) * y / p
    SxyZ = np.cos(thetaPCW(p, c, w))
    # z = np.linspace(0, 4 * np.pi, 500)
    # x = 10 * np.sin(z)
    # y = 10 * np.cos(z)
    ax.plot3D(SXyz, SxYz, SxyZ, linewidth=0.5, color='orangered')
    # ax.plot3D([SXyz,0,0], [0,SxYz,0], [0,0,SxyZ], 'black')
    # ax.plot3D([0,18,0],[5,18,10],[0,5,0],'om-')   #绘制带o折线
    plt.show()