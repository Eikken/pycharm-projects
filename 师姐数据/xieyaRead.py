#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   xieyaRead.py    
@Time    :   2022/3/7 11:39  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import os, glob
import random
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import statsmodels.api as sm
import time

from matplotlib.ticker import MultipleLocator


def draw_line(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    if tM[0][1] > 3000:
        baseLine = tM[0][1] - 3000
        plt.plot(tM[:, 0], tM[:, 1] - baseLine + can, label=fn)


def sort_key(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces


if __name__ == '__main__':
    t1 = time.time()
    filePath = r'C:\Users\Celeste\Desktop\xieyashuju\xieyashuju'  # 文件的根目录
    colorList = ['red', 'DarkOrange', 'Gold', 'green', 'cyan', 'blue', 'purple', 'Chocolate', 'LightPink',
                 'OliveDrab']  # 曲线颜色

    fileList = os.listdir(filePath)
    canshu = 0
    fileList.sort(key=sort_key)

    for fL in fileList:
        f = os.path.join(filePath, fL)
        thisFile = np.array(pd.read_csv(f, header=None))
        # shape0 = thisFile.shape[0]
        shape0 = 537
        thisMatrix = np.zeros((shape0-37, 2))
        for row in range(37, shape0):
            tmp = list(map(float, thisFile[row][0].split(' ')))
            thisMatrix[row-37][0] = tmp[0]
            thisMatrix[row-37][1] = tmp[1]
        draw_line(xy=thisMatrix, c=canshu*3000, fname=fL.split('.')[0])
        canshu += 1
        # break
    plt.xlim(100, 400)
    x_major_locator = MultipleLocator(25)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # ax为两条坐标轴的实例
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    # plt.title('nmm')
    # plt.legend()
    plt.yticks([])
    plt.show()
    t2 = time.time()
    print('finish, use time %d s' % (t2-t1))

