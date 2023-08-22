#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Excel同类型文件合并.py    
@Time    :   2021/10/26 17:02  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import xlrd
import os
import xlsxwriter
import xlwt

source_xls = []
target_xls = "C:/Users/Celeste/Desktop/博导信息采集/A21名博导信息汇总.xlsx"
filePath = "C:/Users/Celeste/Desktop/博导信息采集"
sheetName = ['表1-1-1 博士导师信息（时点）', '表1-1-1博士导师信息（续1）（时点)',
             '表1-1-1 博士导师信息表（续2）（时点）', '表1-1-2 博士生信息表（时期）',
             '表1-2-1 科研项目情况（时期）', '表2-1-1 开课情况（时期）',
             '表2-1-2 研究生教育教学改革研究项目情况（时期）', '表2-1-3 出版教材情况（时期）',
             '表2-1-4 研究生教学成果获奖情况（时期）', '表2-1-5 指导博士生获奖情况（时期）',
             '表2-2-1 科研论文情况（时期）', '表2-2-2 科研获奖情况（时期）', '表2-2-3 出版著作情况（时期）',
             '表2-2-4  专利（著作权）授权情况（时期）', '表2-2-5 参加国际学术会议情况（时期）',
             '表2-2-6 学术任职情况（时点）', '表2-2-7 依托科研平台情况（时点）']
nSheet = {}
allDataDict = {'表1-1-1 博士导师信息（时点）': [],
               '表1-1-1博士导师信息（续1）（时点)': [],
               '表1-1-1 博士导师信息表（续2）（时点）': [],
               '表1-1-2 博士生信息表（时期）': [],
               '表1-2-1 科研项目情况（时期）': [],
               '表2-1-1 开课情况（时期）': [],
               '表2-1-2 研究生教育教学改革研究项目情况（时期）': [],
               '表2-1-3 出版教材情况（时期）': [],
               '表2-1-4 研究生教学成果获奖情况（时期）': [],
               '表2-1-5 指导博士生获奖情况（时期）': [],
               '表2-2-1 科研论文情况（时期）': [],
               '表2-2-2 科研获奖情况（时期）': [],
               '表2-2-3 出版著作情况（时期）': [],
               '表2-2-4  专利（著作权）授权情况（时期）': [],
               '表2-2-5 参加国际学术会议情况（时期）': [],
               '表2-2-6 学术任职情况（时点）': [],
               '表2-2-7 依托科研平台情况（时点）': []}
for file in os.listdir(filePath):
    file_path = os.path.join(filePath, file)
    if '~$' not in os.path.splitext(file_path)[0].split('\\')[1]:
        source_xls.append(file_path)  # 当前待处理文件夹下所有文件路径信息,~$去掉打开文件的本地备份信息

for i in source_xls[:1]:
    wb = xlrd.open_workbook(i)
    for j in sheetName:
        currentSheet = wb.sheet_by_name(j)
        nSheet[j] = [currentSheet.row_values(1)]
# nSheet 存储的内容如下
# [['表1-1-1 博士导师信息（时点）', ['导师唯一识别码', '姓名', '所属学院(单位)', '国家(地区)', '证件类型',
# '证件号码', '出生日期', '性别', '民族', '政治面貌', '最高学历', '获得最高学历的国家(地区)',
# '获得最高学历的院校或机构', '获最高学历所学专业', '最高学位层次', '最高学位名称',
# '获最高学位的国家(地区)', '获最高学位的院校或机构']],

for i in source_xls[2:]:
    wb = xlrd.open_workbook(i)
    for j in nSheet.keys():
        currentSheet = wb.sheet_by_name(j)
        if currentSheet.nrows == 2:
            continue
        else:
            for rownum in range(2, currentSheet.nrows):
                # ce = currentSheet.cell(rownum, 0)
                allDataDict[j].append(currentSheet.row_values(rownum))

xd = xlwt.Workbook()

for k, v in allDataDict.items():
    row = 1
    col = 0
    k_sheet = xd.add_sheet(k)
    for i2 in nSheet[k][0]:  # sheet column
        k_sheet.write(row, col, i2)
        col += 1
    for vv in v:
        row += 1
        col = 0
        for i3 in vv:
            k_sheet.write(row, col, i3)
            col += 1
xd.save(target_xls)
print('finish')
