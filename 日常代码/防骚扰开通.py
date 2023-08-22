#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   防骚扰开通.py    
@Time    :   2022/3/24 18:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import xlwt
import numpy as np
import pandas as pd
import re
from prettytable import PrettyTable


if __name__ == '__main__':
    file1 = r'C:\Users\Celeste\Desktop\开通防骚扰.xlsx'
    file2 = r'C:\Users\Celeste\Desktop\KTFSR.xlsx'

    dataSet1 = np.array(pd.read_excel(file1)['已开通姓名'])
    dataSet2 = np.array(pd.read_excel(file1, sheet_name='Sheet2')[['学号', '姓名', '已开通姓名']])
    # xd = xlwt.Workbook()
    # sheet1 = xd.add_sheet('Sheet1')
    # title = ['学号', '姓名', '已开通姓名']
    # row = 0
    # col = 0
    # for i in title:
    #     sheet1.write(row, col, i)
    #     col += 1
    # for i in dataSet2:
    #     row += 1
    #     col = 0
    #     sheet1.write(row, 0, str(i[0]))
    #     sheet1.write(row, 1, i[1])
    #     if i[1] in dataSet1:
    #         sheet1.write(row, 2, i[1])
    # xd.save(file2)
    table = PrettyTable(['编号', '姓名'])
    n = 1
    # for i in dataSet1[~pd.isna(dataSet1)]:
    #     if i not in dataSet2[:, 1]:
    #         table.add_row([n, i])
    #         n += 1
    # print(table)

    for i in dataSet2[:,1]:
        if i not in dataSet1:
            table.add_row([n, i])
            n += 1
    print(table)
    print('finish')
