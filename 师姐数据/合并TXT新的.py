#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   合并TXT新的.py    
@Time    :   2021/11/28 12:29  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import os
import xlwt


def takeTwo(param):
    # 根据文件名排序
    return int(param.split(' ')[0])


fileDir = r'C:\Users\Celeste\Desktop\dataSet'
fNames = os.listdir(fileDir)
fNames = [i for i in fNames if '.txt' in i]
allList = []
fNames.sort(key=takeTwo)  # 对 file name 排序
book = xlwt.Workbook()  # 创建Excel
sheet = book.add_sheet('Sheet1')
row = 0  # 行
col = 0  # 列
titleList = [i.split('.')[0] for i in fNames[1::2]]
sheet.write(row, col, '0 1 (X-Axis)')
col += 1
for t in titleList:  # 写表头
    sheet.write(row, col, t)
    col += 1

row = 1  # 行加一
col = 0  # 从第0列开始写
for f in fNames:
    print('X-Axis')
    if '0 1 (X-Axis)' in f:  # 写入一列x
        filePath = os.path.join(fileDir, f)
        for line in open(filePath):
            sheet.write(row, col, line)
            row += 1  # 行加一
    break
for f in fNames:  # 写入所有的 y axis
    if 'Y-Axis' in f:
        row = 1  # 行从1开始
        col += 1
        print(f)
        filePath = os.path.join(fileDir, f)
        for line in open(filePath):
            sheet.write(row, col, line)
            row += 1  # 行加一
book.save(os.path.join(fileDir, 'result.xls'))

for i in range(len(titleList)):
    with open(os.path.join(fileDir+'\data\\%s.txt' % titleList[i]), 'w') as w:
        print(titleList[i])
        fx, fy = fNames[i*2], fNames[i*2+1]
        fxPath = os.path.join(fileDir, fx)
        fyPath = os.path.join(fileDir, fy)
        for x, y in zip(open(fxPath), open(fyPath)):
            w.write(x.strip())
            w.write('\t')
            w.write(y.strip())
            w.write('\n')

print('finish')
