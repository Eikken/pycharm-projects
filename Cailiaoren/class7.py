#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class7.py    
@Time    :   2021/3/5 13:11  
@Tips    :   第一性原理软件分子动力学计算结果处理
提取 VASP AIMD结果的能量、温度、坐标数据
基于原子的位置分析O-H键的键长，H-O-H键角的变化趋势
'''
import numpy as np
from matplotlib import pyplot as plt
import os
def lattice_read(XDATCAR7):
    XDATCAR7.readline() # 一行一行的读取
    scaling_factor = float(XDATCAR7.readline())
    lattice = np.zeros((3,3))
    for i in range(3):
        lattice[i] = list(map(float,XDATCAR7.readline().split()))
    # print(lattice)
    element = XDATCAR7.readline().split()
    element_num = list(map(int,XDATCAR7.readline().split()))
    # print(element)
    # print(element_num)
    all_element = []
    for i in range(len(element)):
        for j in range(element_num[i]):
            all_element.append(element[i])
    return scaling_factor,lattice,all_element

def coord_read(XDATCAR7,all_element):
    XDATCAR7.readline()
    atomic_coord = np.zeros((len(all_element), 3))
    for i in range(len(all_element)):
        atomic_coord[i] = list(map(float, XDATCAR7.readline().split()))
    return atomic_coord
step = []
T = []
E = []
with open('data/OSZICAR','r') as reader:
    all_content = reader.readlines()
    for i in all_content:
        if 'T=' in i:
            tmp = i.split()
            step.append(int(tmp[0]))
            T.append(float(tmp[2]))
            E.append(float(tmp[4]))
potim = 0.5
plt.figure(figsize=(16, 16), dpi=300)
plt.subplot(2,2,1) # 两行两列第1个fig
plt.plot(np.array(step)*potim/1000.0,np.array(T))
plt.title('Temperature')
plt.xlabel('Time (fs)')
plt.ylabel('T (k)')
plt.subplot(2,2,2) # 两行两列第2个fig
plt.plot(np.array(step)*potim/1000.0,np.array(E))
plt.title('total energy')
plt.xlabel('Time (fs)')
plt.ylabel('E (eV)')
plt.savefig('png/fig1.png',dpi=300)
# plt.show()
if __name__ == '__main__':
    XDATCAR7 = open('data/XDATCAR7','r')
    scaling_factor,lattice,all_element = lattice_read(XDATCAR7)
    atomic_coord = coord_read()
    tmp = XDATCAR7.readline()
    NPT = True
    if "Direct configuration" in tmp :
        NPT = False
    XDATCAR7.seek(XDATCAR7.tell()-len(tmp),0)
    while XDATCAR7.tell() < os.path.getsize('data/OSZICAR'):
        if NPT == True:
            scaling_factor,lattice,all_element = lattice_read(XDATCAR7)
        current_coord = coord_read(XDATCAR7,all_element)
        print(current_coord)