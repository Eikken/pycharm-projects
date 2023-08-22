#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   公务员报考科目筛选.py    
@Time    :   2022/10/24 22:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    # 设置颜色
style = xlwt.easyxf('pattern: pattern solid, fore_colour ice_blue')
# 字体加粗
style = xlwt.easyxf('font: bold on')
#样式合并
style = xlwt.easyxf('pattern: pattern solid, fore_colour ice_blue; font: bold on')
# 为指定单元格设置样式
sheets.write(0, 0, "hello girl", style)
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xlwt
import os


def in_major(name__):
    for ml in major_list:
        if ml in name__:
            return True
    return False


def isNan(param):
    if pd.isnull(param):
        return True
    if '女性' in param:
        return False
    return True


if __name__ == '__main__':
    # start here
    file1 = r'C:\Users\Celeste\Desktop\jianzhang\2023年度招考简章.xls'
    file2 = r'C:\Users\Celeste\Desktop\jianzhang\材料光电类2023年度招考简章重制.xls'

    # major_list = ['材料类', '材料与化工', '材料科学', '0804', '电子信息', '0807']
    # major_list = ['材料类', '材料与化工', '材料科学', '0804', '软件工程', '电子信息', '计算机类']
    major_list = ['材料类', '材料科学', '光电功能与信息材料', '0804']
    # major_list = ['机械类', '机械工程', '0802']

    f = pd.ExcelFile(file1)
    title = ['部门代码', '部门名称', '用人司局', '机构性质', '招考职位', '职位属性',
             '职位分布', '职位简介', '职位代码', '机构层级', '考试类别', '招考人数',
             '专业', '学历', '学位', '政治面貌', '基层工作最低年限', '服务基层项目工作经历',
             '是否在面试阶段组织专业能力测试', '面试人员比例', '工作地点', '落户地点', '备注',
             '部门网站', '咨询电话1', '咨询电话2', '咨询电话3']

    xd = xlwt.Workbook()
    sheet1 = xd.add_sheet('光电材料类')

    row = 0
    col = 0
    for i in title:
        sheet1.write(row, col, i, style=xlwt.easyxf('font: bold on'))
        col += 1

    for i in f.sheet_names:
        df = pd.DataFrame(pd.read_excel(file1, sheet_name=i))
        c_list = df.values.tolist()[0]  # 得到想要设置为列索引【表头】的某一行提取出来
        df = df[1:]
        df.columns = c_list
        for j, k in zip(df['专业'], df.values):
            if k[16] == '无限制' and k[17] == '无限制' and isNan(k[22]) and k[13] != '仅限本科':
                if in_major(j):
                    row += 1
                    for col in range(len(title)):
                        if pd.isnull(k[col]):
                            sheet1.write(row, col, '-')
                        else:
                            sheet1.write(row, col, k[col])

    if os.path.exists(file2):
        os.remove(file2)
    xd.save(file2)
    print('finish')
