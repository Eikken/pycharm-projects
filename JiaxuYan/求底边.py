#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   求底边.py    
@Time    :   2021/6/30 20:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   a*a=b*b+c*c-2bc*cosA
            顺便学习读写xls
'''
import numpy as np
import pandas as pd
import xlrd
import xlwt
from xlutils.copy import copy


def readData():
    return pd.read_excel(r'data/angle-length.xls')


def getL(a_, theta):
    theta = np.deg2rad(theta)
    return (2 * a_ ** 2 * (1 - np.cos(theta))) ** 0.5


if __name__ == '__main__':
    dataSet = readData()[['angle', 'length']]
    title = 'bottom_edge'
    for k in np.array(dataSet):
        # print(k[1], k[0])
        print(k, ' >> ', getL(k[1], k[0]))
    # read_book = xlrd.open_workbook(r'data/150_角度_边长_胞心距2.xls')
    # # sheet = read_book.sheet_by_index(1)  # 索引的方式，从0开始
    # # sheet.cell(0, 0).value
    # sheet = read_book.sheet_by_name('sheet1')  # 名字的方式
    # max_row = sheet.nrows  # 最大行数
    # max_col = sheet.ncols  # 最大列数
    # newExcel = copy(read_book)
    # ws = newExcel.get_sheet(0)
    # ws.write(0, 3, title)
    # for i in range(1, max_row):
    #     sideEdge = sheet.cell(i, 1).value
    #     angle = sheet.cell(i, 0).value
    #     ws.write(i, 3, getL(sideEdge, angle))
    # newExcel.save('data/newExcel2.xls')
    print('finish')
