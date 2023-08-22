#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   态密度函数计算.py    
@Time    :   2020/12/24 14:18  
@Tips    :
'''

import random,sys,linecache,csv,math,time
import shutil, os
import xml.etree.ElementTree as ET
import numpy as np
import numpy.linalg as la

title = 'test.dens'
GP = 'GP'
data = open('%s' % title)

for n,line in enumerate(data): # print i, seq[i] 0 one,n为下标，line为data
    if n==2:
        freq1 = float(line.split()[0])#第一列的数据
    if n == 3:
        freq2 = float(line.split()[0])
        winSize = (freq2 - freq1)/2
data.close()
data = open('%s' % title) # 防止N=4继续读文件，我们要从头开始读文件。
outPut = open('data\\DoS&gp','w',newline='')
write = csv.writer(outPut)
for n,line in enumerate(data):
    m = 0
    gpList = []
    if n != 0 and n != 1 and n <= 500:
        freq = float(line.encode().split()[0])
        dos = float(line.encode().split()[1])
        floor = freq - winSize
        roof = freq + winSize
        datagp = open('%s' % GP)
        for linegp in datagp:
            m = m + 1
            [freqgp,gp] = linegp.split(',', 2 )
            freqgp = float(freqgp)
            gp = float(gp)
            if freqgp >= floor and freqgp <= roof:
                gpList.append(gp)
        datagp.close()
        if len(gpList) != 0:
            averagegp = math.fsum(gpList)/len(gpList)
        else:
            averagegp = 0
        write.writerow([freq,dos,averagegp])
        # print(freq,dos,averagegp)
outPut.close()
print('结束了')