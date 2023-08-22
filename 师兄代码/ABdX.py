#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   ABdX.py    
@Time    :   2022/6/11 10:50  
@E-mail  :   iamwxyoung@qq.com
@Tips    :  # 一元二次方程组求解 a*t^2 + b*t + c = 0 先判断根有没有
            # 公式化简后有
            # AB^2*t^2 + (1-B^2)*t - A = 0
            # t = e^(-d*X)
            # x = (-1/d)ln(t)
'''

import numpy as np
import pandas as pd
import math
import xlwt


def calculate_x(line_):
    A_ = line_[0]
    B_ = line_[1]
    d_ = line_[2]

    a = A_ * B_ ** 2  # b**2 == b的平方
    b = 1 - B_ ** 2
    c = -A_
    delta_ = b * b - 4 * a * c

    if delta_ < 0:
        return ['', -1]  # 'delta < 0'

    elif delta_ == 0:
        t1_ = -b / (2 * a)
        x1_ = - math.log(t1_ ** (1 / d_))

        return [x1_, 0]  # 'delta == 0'
    else:
        t1_ = (-b + math.sqrt(delta_)) / (2 * a)
        # print(t1_, math.log(t1_), 1 / d_)
        # t2_ = (-b - math.sqrt(delta_)) / (2 * a)
        x1_ = - math.log(t1_ ** (1 / d_))
        # x2_ = - math.log(t2_ ** (1 / d_))
        return [x1_, 1]  # 'delta > 0'


if __name__ == '__main__':
    filePath = r'C:\Users\Celeste\Desktop\ABdX.xlsx'  # 单引号内切换成自己的文件路径
    dataSet = pd.read_excel(filePath)
    titleList = ['A', 'B', 'd', 'X', 'delta']
    # 先处理一下数据，读取出来是文本类型，转化成float才能计算
    dataArr = dataSet.values
    dataLen = len(dataArr)  # 多少行数据

    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('Sheet1')
    row = 0
    col = 0
    for i in titleList:
        sheet1.write(row, col, i)
        col += 1

    for i in range(dataLen):  # 下标索引访问，每一行数据都算一遍,得到x, i也可以当计数器
        col = 0  # 每次循环从第一列开始
        row += 1  # 下一行

        line = dataArr[i]  # 行数据
        res = calculate_x(line_=line)  # 写数据列 res = [x1, x2, x3, ...]   res[0]
        for ABd in line:  # [A, B, d]
            sheet1.write(row, col, ABd)
            col += 1

        for x in res:  # 写结果 [x1, delta]
            sheet1.write(row, col, x)
            col += 1

    xd.save('ABd_x.xls')  #  保存文件
    print('finish')