#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   青年大学习匹配人名.py    
@Time    :   2021/12/20 10:44  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import xlwt
import numpy as np
import pandas as pd
import re


def find_chinese(ss):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', ss)
    for j in sourceSet[:, 0]:
        if j in chinese:
            # print(sourceSet[np.where(sourceSet[:, 0] == j)])
            return j
    return chinese


if __name__ == '__main__':
    file1 = r'D:\Tencent\1748262858\FileRecv\南工大-先进材料研究院(1).xlsx'
    file2 = r'C:\Users\Celeste\Desktop\青年大学习匹配后名单.xlsx'
    zidian = r'D:\Tencent\1748262858\FileRecv\青年大学习人员汇总.xlsx'

    dataSet = np.array(pd.read_excel(file1)[['ID', '姓名', '编号', '归属组织', '报名时间']])
    sourceSet = np.array(pd.read_excel(zidian)[['姓名', '班级']])
    # name = find_chinese('杨晓伟')  # 找到汉字人名部分
    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('Sheet1')
    title = ['ID', '姓名', '编号', '归属组织', '报名时间']
    row = 0
    col = 0
    for i in title:
        sheet1.write(row, col, i)
        col += 1
    for i in dataSet:
        row += 1
        col = 0
        name = find_chinese(i[1])  # 找到汉字人名部分
        sheet1.write(row, 0, i[0])
        sheet1.write(row, 1, name)
        sheet1.write(row, 2, i[2])
        sheet1.write(row, 3, i[3])
        sheet1.write(row, 4, i[4])
    xd.save(file2)
    print('finish')
