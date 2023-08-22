#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   pRaman.py    
@Time    :   2022/3/9 14:56  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


import os, glob
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import statsmodels.api as sm
import time
from matplotlib.ticker import MultipleLocator


def draw_Smooth(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    lowess = sm.nonparametric.lowess
    result = lowess(tM[:, 1], tM[:, 0], frac=0.2, it=3, delta=0.0)
    smoothData[:, can+1] = result[:, 1]
    # plt.plot(result[:, 0], result[:, 1] + can * 50, label=fn)


def draw_line(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    plt.plot(tM[:, 0], tM[:, 1] + can * 700, linewidth=2.5, label=fn)


if __name__ == '__main__':
    t1 = time.time()
    # filePath = r'C:\Users\Celeste\Desktop\praman'  # 文件的根目录
    filePath = r'C:\Users\Celeste\Desktop\vraman'  # 文件的根目录
    fNames = os.listdir(filePath)
    xyFileName = []
    for i in range(13):
        angle = i * 15
        xyFileName.append(['%d 1 (X-Axis).txt' % angle, '%d 1 (Y-Axis).txt' % angle])
    dataSet = np.zeros((1024, 26))
    smoothData = np.zeros((1024, 13))
    inDx = 0
    for fN in xyFileName:
        txt = os.path.join(filePath, fN[0])
        tyt = os.path.join(filePath, fN[1])
        row = 0
        for line in open(txt):
            dataSet[row, 2 * inDx] = float(line.split()[0])
            row += 1
        row = 0
        for line in open(tyt):
            dataSet[row, 2 * inDx + 1] = float(line.split()[0])
            row += 1
        inDx += 1
    # 现在dataSet里存的是[x1,y1,x2,y2...]
    config = {
        # "font.family": 'Times New Roman',
        "font.size": 20,  # 设置字体类型
    }
    plt.rcParams.update(config)
    plt.figure(figsize=(6, 9))
    smoothData[:, 0] = dataSet[:, 0]
    for i in range(13):
        thisMatrix = dataSet[:, 2 * i: 2 * i + 2]
        draw_line(xy=thisMatrix, c=i, fname=str(i * 15))
    # pd.DataFrame(smoothData).to_excel('apl.xls', index=False)
    plt.yticks([])
    plt.xlim(100, 500)
    # plt.title('pRaman')
    # plt.legend(loc='upper right')
    plt.show()
    print('finish')
