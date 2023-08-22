#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   画十条线.py    
@Time    :   2021/3/8 15:26  
@Tips    :   读取x，y坐标，绘制曲线
'''

import os, glob
import random
import numpy as np
from matplotlib import pyplot as plt


def quChuWuXiaoFile(allFileList):
    newFileList = []
    for i in allFileList:
        if 'Header' in i:
            continue
        else:
            newFileList.append(i)
    return newFileList


def getXY_axis(filePath, formatStr):  # 格式 '01 60-0pl (Y-Axis).txt'>>60
    return glob.glob(filePath + '\\' + formatStr)


def readXYList(axisFilePath):  # 传进来一个文件名列表，读取所有文件的坐标值
    newList = np.zeros((len(axisFilePath), 1024))  # np.array(len(axisFilePath)) 一共 len（）行，1024列
    for i in range(len(axisFilePath)):
        newList[i] = np.loadtxt(axisFilePath[i])
    return newList
    # return pd.read_csv(axisFilePath[0])


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def draw_60_func(x_axis, y_axis):
    for i in range(10):  # +(i*5000)
        canshu = i * 5000
        if i == 1:  # 优化2线条
            canshu = 2 * i * 5000
            plt.plot(x_axis[i], (y_axis[i] + canshu) * 0.5, color=colorList[i], linewidth=0.5, label='line-' + str(i))
        elif i == 5 or i == 6:  # 优化6、7线条
            canshu = i * 5000 * 0.5
            plt.plot(x_axis[i], (y_axis[i] + canshu) * 2, color=colorList[i], linewidth=0.5, label='line-' + str(i))
        else:
            plt.plot(x_axis[i], y_axis[i] + canshu, color=colorList[i], linewidth=0.5, label='line-' + str(i))


def draw_0_func(x_axis, y_axis):
    for i in range(10):
        canshu = i * 5000
        if i == 1:  # 优化2线条
            canshu = 1.7 * i * 5000
            plt.plot(x_axis[i], (y_axis[i] + canshu) * 0.6, color=colorList[i], linewidth=0.5, label='line-' + str(i))
        else:
            plt.plot(x_axis[i], y_axis[i] + canshu, color=colorList[i], linewidth=0.5, label='line-' + str(i))


if __name__ == '__main__':
    filePath = r'C:\Users\Celeste\Desktop\PL'  # 文件的根目录
    colorList = ['red', 'DarkOrange', 'Gold', 'green', 'cyan', 'blue', 'purple', 'Chocolate', 'LightPink',
                 'OliveDrab']  # 曲线颜色
    allFileList = quChuWuXiaoFile(os.listdir(filePath))  # 读取出文件中的文件名，存在list中
    # 取0的 1*0-*(Y-Axis).txt   *为通配符
    xName = '*0-*(X-Axis).txt'
    yName = '*0-*(Y-Axis).txt'
    x_axisFile = getXY_axis(filePath, xName)[10:]
    y_axisFile = getXY_axis(filePath, yName)[10:]  # 0-19 一共20组xy，[:10]是前十组(60度)文件,[10:]是后10个（0度），按x y区分
    # 所有你需要修改的除了上边的[:10]还是[:10],还有下面的调用函数
    x_axis = readXYList(x_axisFile)
    y_axis = readXYList(y_axisFile)

    # 调用函数，60调60的，0调0的
    # draw_0_func(x_axis,y_axis)
    draw_60_func(x_axis, y_axis)

    # plt.grid() #绘制方格
    plt.yticks([])  # 不显示y坐标值
    plt.legend(loc=1)  # 设置图例的位置'upper right', 'upper left', 'lower left', 'lower right',
    # 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    plt.savefig('png/十条曲线.png', dpi=1000)  # 图片存在png文件夹下
    plt.show()
    print('finish')
