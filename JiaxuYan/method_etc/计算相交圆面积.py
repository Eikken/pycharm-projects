#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   计算相交圆面积.py    
@Time    :   2021/3/15 21:38  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   解决了公式法求圆相交部分的面积，且与积分的数值相同
'''
import numpy as np
from matplotlib import pyplot as plt


class Circle:
    x = 0.0
    y = 0.0
    r = 1.0

    def __init__(self, x, y, r):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)

    def calArea(self):
        return np.pi * self.r ** 2


def draw(c_1, c_2):
    x_1 = np.linspace(c_1.x - c_1.r, c_1.x + c_1.r, 200)
    x_2 = np.linspace(c_2.x - c_2.r, c_2.x + c_2.r, 200)
    y_1 = np.sqrt(c_1.r ** 2 - (x_1 - c_1.x) ** 2)
    y_2 = np.sqrt(c_2.r ** 2 - (x_2 - c_2.x) ** 2)
    fig = plt.figure(figsize=(6, 4), dpi=100)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_1, y_1, color='blue')
    ax1.plot(x_1, -y_1, color='blue')
    ax1.plot(x_2, y_2, color='red')
    ax1.plot(x_2, -y_2, color='red')
    plt.axhline(y=0, xmin=-10, xmax=10, linestyle='--', color='grey', lw=0.5)
    # plt.axvline(x=0.5, ymin=-10, ymax=10, linestyle='--', color='grey', lw=0.5)
    plt.axvline(x=0, ymin=-10, ymax=10, linestyle='--', color='grey', lw=0.5)
    # plt.axvline(x=1, ymin=-10, ymax=10, linestyle='--', color='grey', lw=0.5)
    # c1 = plt.Circle((0,0),1)
    # plt.gcf().gca().add_artist(c1)
    plt.show()


def calShadow(circle1, circle2):
    d = np.sqrt((circle1.x - circle2.x) ** 2 + (circle1.y - circle2.y) ** 2)
    if d >= circle1.r + circle2.r:
        return '两圆不相交！'
    # # 透镜面积公式 这是个求R的式子
    # area = circle1.r ** 2 / np.cos((d ** 2 + circle1.r ** 2 - circle2.r ** 2) / 2 * d * circle1.r) + \
    #        circle2.r ** 2 / np.cos((d ** 2 - circle1.r ** 2 + circle2.r ** 2) / 2 * d * circle2.r) - \
    #        np.sqrt((d + circle1.r - circle2.r) * (d - circle1.r + circle2.r) * (-d + circle1.r + circle2.r) * (
    #                d + circle1.r + circle2.r)) / 2

    elif d == 0.0:
        return '两圆重合，面积为：%f' % (np.pi * circle1.r ** 2)
    print('圆心距', d, '小于半径 %f + %f' % (circle1.r, circle2.r))
    # 公式法，可用积分法，上下两个曲线积分，积分区间不好定义,积分验证正确
    ang1 = np.arccos((circle1.r ** 2 + d ** 2 - circle2.r ** 2) / 2.0 / circle1.r / d)
    ang2 = np.arccos((-circle1.r ** 2 + d ** 2 + circle2.r ** 2) / 2.0 / circle2.r / d)
    area = ang1 * circle1.r ** 2 + ang2 * circle2.r ** 2 - d * circle1.r * np.sin(ang1)
    return area

def calDistance(c_1,c_2):
    return ((c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2) ** 0.5


while True:
    x, y, r = map(float, input('输入第一个圆的圆心x、y坐标及半径r:').split())
    c1 = Circle(x, y, r)
    x2, y2, r2 = map(float, input('输入第二个圆的圆心x、y坐标及半径r:').split())
    c2 = Circle(x2, y2, r2)
    dis = calDistance(c1,c2)
    print(dis)
    print()
    # draw(c_1=c1, c_2=c2)
