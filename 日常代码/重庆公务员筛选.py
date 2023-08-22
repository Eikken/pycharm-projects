#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   重庆公务员筛选.py    
@Time    :   2023/1/10 16:58  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlwt
import unfolding


def in_major(name__):
    for ml in major_list:
        if ml in name__:
            return True
    return False


if __name__ == '__main__':
    # start here
    file1 = r'data\重庆市2023年度公务员招考职位情况一览表.xlsx'
    file2 = r'data\重制.xls'

    major_list = ['材料', '工学', '光学', '不限']
    f = pd.DataFrame(pd.read_excel(file1))
    title = list(f.loc[1, :])
    f = f[1:]
    f.columns = title

    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('材料类')

    row = 0
    col = 0
    for i in title:
        sheet1.write(row, col, i, style=xlwt.easyxf('font: bold on'))
        col += 1
    for i in f.values:
        if '硕士' in i[8] and '应届' in i[18]:

            if in_major(i[9]):
                print(i)