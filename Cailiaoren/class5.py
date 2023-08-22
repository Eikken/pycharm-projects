#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class5.py    
@Time    :   2021/3/3 14:00  
@Tips    :   态密度的意义 density of states
'''
import numpy as np
file1 = open('data/new2.dat','r')

all_data = file1.read()
file1.seek(0) #文件指针移动到前端
all_data_lines = file1.readlines()
all_data_dos = all_data.split("\n\n")
allLines = []
for i in all_data_dos:
    eachLine = []
    if i.strip('\n') != '':
        # print(i) # 301个点作为 str 打印出来
        lines = i.split('\n')# str行切割
        for line in lines:
            eachLine.append(list(map(float,line.split()))) #类型转换
        allLines.append(eachLine)
for i in allLines[1:]:
    for index in range(len(i)):
        allLines[0][index].append(i[index][1])


with open('data/reform.dat','w') as writer:
    for i in allLines[0]:
        writer.write('%7.3f %7.3f %7.3f\n'%(i[0],i[1],i[2]))

new_data = np.array(allLines[0])
energy = new_data[:,0]
state = new_data[:,2]
# 公式积分
fenzi = np.trapz(energy*state,energy)
fenmu = np.trapz(state,energy) # trapz是梯形积分
dBand = fenzi/float(fenmu)

print(dBand)
