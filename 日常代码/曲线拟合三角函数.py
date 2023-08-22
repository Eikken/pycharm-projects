#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   曲线拟合三角函数.py    
@Time    :   2021/10/31 17:50  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
from scipy.optimize import leastsq
from scipy.optimize import curve_fit


def drawFig(x, y):
    fig = plt.figure(figsize=(8, 8))
    # 使用axisartist.Subplot方法创建一个绘图区对象ax
    ax = axisartist.Subplot(fig, 111)
    # 将绘图区对象添加到画布中
    fig.add_axes(ax)
    ax.axis[:].set_visible(False)  # 通过set_visible方法设置绘图区所有坐标轴隐藏
    ax.axis["x"] = ax.new_floating_axis(0, 0)  # ax.new_floating_axis代表添加新的坐标轴
    ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加上箭头
    # 添加y坐标轴，且加上箭头
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    ax.axis["y"].set_axisline_style("-|>", size=1.0)
    # 设置x、y轴上刻度显示方向
    ax.axis["x"].set_axis_direction("top")
    ax.axis["y"].set_axis_direction("right")
    plt.xlim(-10, 80)
    plt.ylim(-0.5, 0.4)
    plt.scatter(xL, y, marker='o', color='green')
    tmpX = np.linspace(0,75,750)
    tmpTheta = np.deg2rad(tmpX)
    plt.plot(tmpX, func(tmpTheta, *popt), 'r-',color='red')
    # plt.plot(param1)
    plt.show()


def func(x, a, b, c, d, e):
    return a * np.sin(x+b) + c*np.sin(2*x + d) + e


if __name__ == '__main__':
    xL = [0, 15, 30, 45, 60, 75]
    yL = [0.05, 0.03, 0.13, -0.01, -0.46, -0.21]
    tolerance = [0.08, 0.02, 0.06, 0.06, 0.18, 0.03]
    thetaX = np.deg2rad(xL)
    popt, pcov = curve_fit(func, thetaX, yL)
    print(popt)
    drawFig(thetaX, yL, )
    print('finish')
