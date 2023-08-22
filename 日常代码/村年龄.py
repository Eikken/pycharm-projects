#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   村年龄.py    
@Time    :   2022/1/10 15:37  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


import xlwt
import numpy as np
import pandas as pd
import re


def find_age(i):
    name = i[1]
    age = -1
    for j in st2:
        if j[2] == name:
            age = j[6]
            return age
    return age


if __name__ == '__main__':
    file1 = r'C:\Users\Celeste\Desktop\历年硕博导名单---年龄(1).xlsx'
    file2 = r'C:\Users\Celeste\Desktop\历年硕博导名单.xlsx'

    st1 = np.array(pd.read_excel(file1, sheet_name='Sheet1'))
    st2 = np.array(pd.read_excel(file1, sheet_name='Sheet2'))
    columns = pd.read_excel(file1, sheet_name='Sheet1').columns
    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('Sheet1')
    title = columns
    row = 0
    col = 0
    for i in title:
        sheet1.write(row, col, i)
        col += 1
    for i in st1:
        row += 1
        col = 0
        age = find_age(i)  # 找到age
        for k in range(6):
            sheet1.write(row, k, i[k])
        sheet1.write(row, 6, i[6])
        sheet1.write(row, 7, age)
    xd.save(file2)
    print('finish')
    # for i in st1: