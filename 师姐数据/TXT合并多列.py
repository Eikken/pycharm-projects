#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   TXT合并多列.py    
@Time    :   2021/1/29 18:36  
@Tips    :   
'''

import os.path
import copy

def quHuanHang(oldList):
    newL = []
    for i in oldList:
        s = [i.strip() for i in i]
        newL.append(s)
    return newL


cwd = os.getcwd()

filedir = os.path.join(cwd, 'txtdata/123/垂直极化')

fNames = os.listdir(filedir)
# print(fNames)

listAll = []
allL = []
for f in fNames:
    if '.txt' in f:
        filePath = os.path.join(filedir, f)
        tmpList = []
        for line in open(filePath):
            if 'X' in line.split() or len(line.split()) == 1:
                continue
            tmpList.append(line.split()[0])
        listAll.append(tmpList)
        tmpList = []
    break
allL = copy.deepcopy(listAll)
listAll = []
for f in fNames:
    if '.txt' in f:
        filePath = os.path.join(filedir, f)
        tmpList = []
        for line in open(filePath):
            if 'X' in line.split() or len(line.split()) < 2:
                continue
            else:
                tmpList.append(line.split()[1])
        listAll.append(tmpList)
        tmpList = []
for l in listAll:
    allL.append(l)
newList = quHuanHang(allL)
reverseList = list(map(list, zip(*newList)))
with open('../师姐数据/result.txt', 'w') as f:
    for i in range(len(reverseList)):
        for j in reverseList[i]:
            f.write(j)
            f.write('\t')
        f.write('\n')

# for f in fNames:
#     if '.txt' in f:
#         filePath = os.path.join(filedir, f)
#         tmpList = []
#         for line in open(filePath):
#             if 'X' in line.split() or len(line.split()) < 2:
#                 continue
#             else:
#                 tmpList.append(line.split()[1])
#         listAll.append(tmpList)
#         tmpList.clear()
# newList = quHuanHang(listAll)
# # 不去换行，写入文件的时候后自动换行，导致跳行，也可以在写入文件的时候边去边写
# reverseList = list(map(list, zip(*newList)))  # 转置一下变成要写入文件中的格式
# with open('../师姐数据/result.txt', 'w') as f:
#     for i in range(len(reverseList)):
#         for j in reverseList[i]:
#             f.write(j)
#             f.write('\t')
#         f.write('\n')

print('finish')
