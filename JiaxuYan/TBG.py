#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   TBG.py    
@Time    :   2021/7/11 12:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   使用文献中的公式计算super lattice L的大小
'''

import numpy as np
import pandas as pd
import xlrd
import xlwt


def getLm(a_, theta_):
    theta_ = np.deg2rad(theta_)
    return a_ /(2 * (np.sin(theta_ / 2.0)))


if __name__ == '__main__':
    cellLength = {'3.48': 2338.07, '3.89': 2092.21, '4.41': 1846.37, '5.09': 1600.58, '5.36': 2630.40,
                  '6.01': 1354.86, '6.40': 2204.87, '6.84': 2063.08, '7.34': 1109.28, '7.93': 1779.61,
                  '8.61': 1637.95, '9.43': 863.92, '10.42': 1354.86, '11.64': 1213.49, '13.17': 619.09,
                  '15.18': 931.34, '16.43': 994.20, '17.9': 790.78, '21.79': 375.77, '24.43': 1162.55,
                  '26.01': 1262.37, '27.8': 512.09, '29.41': 1398.82}
    a = 142  # 246
    title = ['angle', 'local', 'equation']
    # print(k, 60 - float(k))
    # for k, v in cellLength.items():
        # print(k, '>> local=', v, ' ; equation=', getLm(a, float(k)))
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1')
    row = 0  # 行
    col = 0  # 列
    for t in title:
        sheet.write(row, col, t)
        col += 1
    for k, v in cellLength.items():
        # print(k, '>> local=', v, ' ; equation=', getLm(a, float(k)))
        row += 1
        print(row, '/', len(cellLength))
        col = 0
        contents = [k, v, getLm(a, float(k))]
        for j in contents:
            sheet.write(row, col, j)
            col += 1
    book.save('data/localEquation142.xls')
    print('saved data/localEquation.xls')