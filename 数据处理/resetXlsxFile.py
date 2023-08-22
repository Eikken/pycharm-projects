#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   resetXlsxFile.py    
@Time    :   2021/1/15 15:59  
@Tips    :   整合数据.xls 文件即只有一个col表头的空文件，xlrd需要借助这个空文件来copy生成的文件并且保存
'''

# name = ZHL2-2117-09-NBGC(A6)-GDHT-RTHFD-001-91    2021.1.8日之前(1).xlsx
import xlrd
import os
from xlutils.copy import copy
import pandas as pd
file = r'ZHL2-2117-09-NBGC(A6)-GDHT-RTHFD-001-91    2021.1.8日之前(1).xlsx'
col = ['管线编号/焊口编号', '焊工号', '拍片张数', '不合格情况', '结论', '委托单号', '回复单号', '检测日期']
fr = xlrd.open_workbook(file)
# sheet1 = fr.sheets()[0]
# col = sheet1.row_values(0) # col
# 所有sheet的name>> fr.sheet_names()
dir = os.path.abspath('.').split('src')[0]
oldWb = xlrd.open_workbook(dir + "\\整合数据.xls")
w = copy(oldWb)
r = 1 # 所有行

for f in fr.sheets():
    if f.nrows <= 1:
        pass
    else:
        danhao = f.cell_value(1, 2)#f.row_values(1)
        date = xlrd.xldate_as_datetime(f.cell_value(2, 11), 0).strftime('%Y/%m/%d')
        for row in range(10,f.nrows): # 获取有效的行
            value = f.row_values(row)
            if '以下空白' in value:
                break
            value = [i for i in value if i != '']
            value.append(danhao)
            value.append(date)
            for i in range(len(col)):
                w.get_sheet(0).write(r,i,value[i+1])
            r += 1
w.save('整合数据1.xls')
print('file saved')

