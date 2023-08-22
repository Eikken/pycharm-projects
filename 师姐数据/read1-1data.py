#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   read1-1data.py    
@Time    :   2020/12/25 12:38  
@Tips    :    
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def draw(x,y,points):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(6, 5))
    # plt.scatter(x, y, color='chocolate', linewidths=0.1)
    # plt.scatter(points[:,0],points[:,1], color='limegreen', marker='*')
    plt.plot(x,y,color='HotPink',linewidth=1)
    plt.title('图1-1')
    plt.xlabel('X-aixs')
    plt.ylabel('Y-aixs')
    # plt.savefig('消费信息图')
    # x >> [0,100] , y >> [0,800]
    plt.show()
    print('画完了')
xLable = pd.read_csv(r'txtdata\P1 1 (X-Axis).txt')
yLable = pd.read_csv(r'txtdata\P1 1 (Y-Axis).txt')
points = np.array([[xLable, yLable]])
draw(xLable,yLable,points)
# print(xLable[:10])