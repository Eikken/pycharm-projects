#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   pl2.py    
@Time    :   2022/3/8 11:03  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


import os, glob
import random
import numpy as np
import pandas as pd
import re

import xlwt
from matplotlib import pyplot as plt
import statsmodels.api as sm
import time
from matplotlib.ticker import MultipleLocator


def saveExcel(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    if canshu == 8:
        lowess = sm.nonparametric.lowess
        result = lowess(tM[:, 1] - 1000, tM[:, 0], frac=0.2, it=3, delta=0.0)
        # index = np.where(result[:, 1] == result[:, 1].min())
        dataSet[:, 0] = 1240/result[:, 0]
        dataSet[:, 9] = result[:, 1]
    else:
        lowess = sm.nonparametric.lowess
        result = lowess(tM[:, 1], tM[:, 0], frac=0.2, it=3, delta=0.0)
        dataSet[:, can+1] = result[:, 1]
        # print(result)
        # index = np.where(result[sta:, 1] == result[sta:, 1].max())
        # plt.plot(1240 / result[:, 0], result[:, 1] + can, label=fn)


def draw_line(**kwargs):
    tM = kwargs['xy']
    can = kwargs['c']
    fn = kwargs['fname']
    # plt.plot(tM[:, 0], tM[:, 1] + can, label=fn)
    if canshu == 8:
        lowess = sm.nonparametric.lowess
        result = lowess(tM[:, 1]-1000, tM[:, 0], frac=0.2, it=3, delta=0.0)
        plt.plot(1240 / result[:, 0], result[:, 1] + can*220, label=fn)
    else:
        lowess = sm.nonparametric.lowess
        result = lowess(tM[:, 1], tM[:, 0], frac=0.2, it=3, delta=0.0)
        # print(result)
        # sta = 2400
        # index = np.where(result[sta:, 1] == result[sta:, 1].max())
        plt.plot(1240/result[:, 0], result[:, 1] + can*220, label=fn)
        # plt.scatter(result[sta:][index][:, 0], result[sta:][index][:, 1] + can, color='black')


def sort_key(s):
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces


if __name__ == '__main__':
    t1 = time.time()
    filePath = r'C:\Users\Celeste\Desktop\pl2'  # 文件的根目录
    file2 = r'C:\Users\Celeste\Desktop\pl.xlsx'
    fileList = os.listdir(filePath)
    canshu = 0
    fileList.sort(key=sort_key)
    # plt.figure(figsize=(4, 5))
    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('Sheet1')
    row = 0
    col = 0
    sheet1.write(row, 0, 'tt')
    xd.save(file2)
    dataSet = np.zeros((7360, 10))
    for fL in fileList:
        f = os.path.join(filePath, fL)
        thisFile = np.array(pd.read_csv(f, header=None))
        shape0 = thisFile.shape[0]
        thisMatrix = np.zeros((shape0, 2))
        for row in range(shape0):
            tmp = list(map(float, thisFile[row][0].split(' ')))
            thisMatrix[row][0] = tmp[0]
            thisMatrix[row][1] = tmp[1]
        # draw_line(xy=thisMatrix, c=canshu*220, fname=fL.split('.')[0])
        saveExcel(xy=thisMatrix, c=canshu, fname=fL.split('.')[0])
        canshu += 1
    # print(dataSet[:10])
    pd.DataFrame(dataSet).to_excel('pl2.xls', index=False)
    # plt.xlim(1.48, 1.625)
    # x_major_locator = MultipleLocator(25)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # ax为两条坐标轴的实例
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.yticks([])
    # plt.show()
    t2 = time.time()
    print('finish, use time %d s' % (t2-t1))
