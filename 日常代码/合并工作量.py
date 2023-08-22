#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   合并工作量.py    
@Time    :   2021/12/13 16:15  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import xlrd
import os
import xlsxwriter
import xlwt
import numpy as np
import pandas as pd


if __name__ == '__main__':
    file1 = r'C:\Users\Celeste\Desktop\一二学期工作量汇总表.xlsx'
    file2 = r'C:\Users\Celeste\Desktop\teacher.xlsx'

    dataSecond = np.array(pd.read_excel(file1, sheet_name='第二学期'))
    dataFirst = np.array(pd.read_excel(file1, sheet_name='第一学期'))
    dataResult = np.array(pd.read_excel(file2)[['序号',	'姓名', '工号', '专业技术职务']])
    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('Sheet1')
    title = ['序号', '姓名', '工号', '专业技术职务', '二学期', '一学期']
    row = 0
    col = 0
    for i in title:
        sheet1.write(row, col, i)
        col += 1
    for i in dataFirst:
        if i[0] not in dataResult:
            print('dataFirst',i)
    for i in dataSecond:
        if i[0] not in dataResult:
            print('dataSecond',i)
    # for teacher in dataResult:
    #     if teacher[1] not in dataSecond and teacher[1] not in dataFirst:
    #         continue
    #     row += 1
    #     col = 0
    #     print(row)
    #     sheet1.write(row, col, row)
    #     col += 1
    #     for i in teacher[1:]:
    #         sheet1.write(row, col, i)
    #         col += 1
    #     if teacher[1] in dataSecond:
    #         col = 4
    #         info = dataSecond[np.where(dataSecond==teacher[1])[0]]
    #         secondVal = info[0][-1]
    #         sheet1.write(row, col, secondVal)
    #         # sheet1.write  # [['卢晓梅' 57.6]]
    #     if teacher[1] in dataFirst:
    #         col = 5
    #         info = dataFirst[np.where(dataFirst==teacher[1])[0]]
    #         firstVal = info[0][-1]
    #         sheet1.write(row, col, firstVal)
    #         # sheet1.write  # [['卢晓梅' 57.6]]
    # xd.save('data/teacher汇总.xls')


