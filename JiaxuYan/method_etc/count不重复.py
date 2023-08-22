#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   count不重复.py    
@Time    :   2021/5/19 10:23  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import pandas as pd
import numpy as np
import math
import xlwt

cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '13.17': 619.0864237,
              '15.18': 931.3409687, '16.43': 994.1971635, '17.9': 790.7793624, '21.79': 375.771207,
              '24.43': 1162.550644, '27.8': 512.0898359}
book = xlwt.Workbook()  # 创建Excel
title = ['angle', 'sin', 'cos', 'sinh', 'cosh', 'tan', 'cot']
sheet = book.add_sheet('sheet1')
row = 0  # 行
col = 0  # 列
for t in title:
    sheet.write(row, col, t)
    col += 1
row += 1
col = 0
for k in cellLength.keys():
    theta = np.deg2rad(float(k))
    thetaList = [float(k), np.sin(theta), np.cos(theta), np.sinh(theta), np.cosh(theta), np.tan(theta), 1 / np.tan(theta)]
    for i in thetaList:
        sheet.write(row, col, i)
        col += 1
    col = 0
    row += 1
print('finish')
book.save('data/三角函数值.xls')

# fileName = r'data/6.01df1.xls'
# dataSet = pd.read_excel(fileName)[0]
# tmplist = []
# for v in dataSet.values:
#     tmplist.append(float(str(v).split('.')[0]))
# tmpdict = {}
# for i in tmplist:
#     if i in tmpdict:
#         tmpdict[i] += 1
#     else:
#         tmpdict[i] = 1

# print(tmpdict)
# print(1.0==int(1.111))
