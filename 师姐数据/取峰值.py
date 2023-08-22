#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   取峰值.py    
@Time    :   2021/3/16 21:49  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import os, glob
import random
import numpy as np
from matplotlib import pyplot as plt
import peakutils


def reNameFile():
    PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
    DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'txtdata/123/垂直极化')
    files = os.listdir(DIR_PATH)
    #
    # def is_suffix_txt(suffix: str):
    #     if suffix == 'txt':
    #         return True
    #     return True

    for filename in files:
        nameList = filename.split('-')
        new_name = os.path.join(DIR_PATH, nameList[2])
        old_name = os.path.join(DIR_PATH, filename)
        os.rename(old_name, new_name)
    # print(os.listdir(DIR_PATH))


def delInvalidFile(allFileList):
    newFileList = []
    for i in allFileList:
        if 'sub' in i:
            continue
        else:
            newFileList.append(i)
    return newFileList


def getGlob(filePath, formatStr):
    return sorted(glob.glob(filePath + '\\' + formatStr), key=os.path.getmtime)  # 按时间排序终于成了


def readXYList(axisFilePath, index):  # 传进来一个文件名列表，读取所有文件的坐标值
    newList = np.zeros((25, 1259))  # np.array(len(axisFilePath)) 一共 len（）行，1
    for i in range(len(axisFilePath)):
        newList[i] = np.loadtxt(axisFilePath[i], skiprows=2, usecols=(index))
    return newList
    # return pd.read_csv(axisFilePath[0])


def randomRainbow():
    colorList = ['red', 'DarkOrange', 'Gold', 'green', 'cyan', 'blue', 'purple', 'Chocolate', 'LightPink', 'OliveDrab',
                 'DarkSlateBlue', 'DeepPink', 'BlueViolet', 'ForestGreen', 'Olive', 'Sienna', 'Maroon']
    i = random.randint(0, len(colorList))
    return colorList[i]


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


# reNameFile() # 改完了

allFileList = getGlob(os.path.join(os.getcwd(), 'txtdata/123/垂直极化'), '*.txt')
allFileList = delInvalidFile(allFileList)
xList = readXYList(allFileList, 0)
yList = readXYList(allFileList, 1)

while True:
    num = int(input('0-25组数据：'))
    if num > 25 or num < 0:
        print('ERROR：数据下标越界！')
        continue
    else:
        indexes = peakutils.indexes(yList[num], thres=0.2, min_dist=30)
        print(xList[num][indexes], yList[num][indexes])
        plt.plot(xList[num], yList[num], color=randomRainbow(), lw='0.5')
        plt.scatter(xList[num][indexes], yList[num][indexes], marker='.', color='black', lw=0.3)
        plt.savefig('拉曼偏角%d°.png'%(num*15), dpi=300)
        plt.show()
