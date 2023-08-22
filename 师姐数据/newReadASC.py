#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   newReadASC.py    
@Time    :   2022/1/22 14:23  
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


def draw_line(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    lowess = sm.nonparametric.lowess
    result = lowess(tM[:, 1], tM[:, 0], frac=0.2, it=3, delta=0.0)
    # print(result)
    sta = 2400
    index = np.where(result[sta:, 1] == result[sta:, 1].max())
    plt.plot(result[:, 0], result[:, 1]+can, label=fn)
    plt.scatter(result[sta:][index][:, 0], result[sta:][index][:, 1]+can, marker='+', color='black')


def sort_key(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces


if __name__ == '__main__':
    t1 = time.time()
    filePath = r'C:\Users\Celeste\Desktop\pl'  # 文件的根目录
    colorList = ['red', 'DarkOrange', 'Gold', 'green', 'cyan', 'blue', 'purple', 'Chocolate', 'LightPink',
                 'OliveDrab']  # 曲线颜色

    fileList = os.listdir(filePath)
    canshu = 0
    fileList.sort(key=sort_key)
    for fL in fileList:
        f = os.path.join(filePath, fL)
        thisFile = np.array(pd.read_csv(f, header=None))
        shape0 = thisFile.shape[0]
        thisMatrix = np.zeros((shape0, 2))
        for row in range(shape0):
            tmp = list(map(float, thisFile[row][0].split(' ')))
            thisMatrix[row][0] = tmp[0]
            thisMatrix[row][1] = tmp[1]
        draw_line(xy=thisMatrix, c=canshu*220, fname=fL.split('.')[0])
        canshu += 1
        # break
    plt.title('nmm')
    # plt.legend()
    plt.show()
    t2 = time.time()
    print('finish, use time %d s' % (t2-t1))

