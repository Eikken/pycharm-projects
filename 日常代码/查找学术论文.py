#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   查找学术论文.py    
@Time    :   2022/1/10 17:15  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import xlwt
import numpy as np
import pandas as pd
import re


if __name__ == '__main__':
    file1 = r'E:\桌面文件备份\杨丽娟老师\石弘颖老师\学位点IAM人员信息表.xlsx'
    file2 = r'E:\桌面文件备份\杨丽娟老师\20220110\2021论文-20211223整理.xlsx'
    file3 = r'C:\Users\Celeste\Desktop\论文筛查汇总.xlsx'
    # 查询 file2中有file1 teacher name的信息
    st1 = np.array(pd.read_excel(file1)['姓名'])  # all name in here
    st2 = np.array(pd.read_excel(file2, sheet_name='1223整理'))
    columns = pd.read_excel(file2, sheet_name='1223整理').columns
    # xd = xlwt.Workbook()
    # sheet1 = xd.add_sheet('Sheet1')
    # title = columns
    # row = 0
    # col = 0
    # for i in title:
    #     sheet1.write(row, col, i)
    #     col += 1
    for info in st2[2:3]:
        for i in range(len(info)):
            print(i, info[i])
        # for n in nameList:

            # if n in st1:
            #     row += 1
            #     for col in range(12):
            #         sheet1.write(row, col, info[col])
    # xd.save(file3)

    print('finish')