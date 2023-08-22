#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   xmlRead.py    
@Time    :   2020/12/27 14:26  
@Tips    :   读取获取能带、费米啥的
'''
import random,sys,linecache,csv,math,time
import shutil, os
import numpy as np
import xml.etree.ElementTree as et

# 读取信息
nkp = 194 # k-points 能带计算
skp = 145 # 其实k-points
nBand = 64 # 能带数
nGrid = 301 # grid point 电子态密度
switch = 1 # 费米能级 0 关  1 开
tree = et.parse('data/vasprun.xml')
root = tree.getroot()

outPut = open('data/atomlist','w')
nAtom = 0
for rc in root.findall("./atominfo/array[@name='atoms']/set/rc"):
    atName = rc[0].text
    nAtom += 1
    outPut.write(str(nAtom)+' '+atName+'\n')
outPut.close()

outPut = open('data/kplist','w')
nkPoint = 0
for v in root.findall("./kpoints/varray[@name='kpointlist']/v"):
    kpoint = v.text
    nkPoint += 1
    outPut.write(str(nkPoint)+kpoint+'\n')
outPut.close()
# 读取费米能级
for i in root.findall("./calculation/dos//i[@name='efermi']"):
    efermi = float(i.text) * switch
    print('Fermi level:',str(efermi))

outPut = open('data/band','w',newline='')
write = csv.writer(outPut,delimiter=' ')
arrband = []
#//*[@id="folder1269"]/div[1]
for kpNum in range(2,69):
    for r in root.findall("./calculation/eigenvalues/array//set[@comment='spin 1']/set[@comment='kpoint %d']/r" % (kpNum)):
        [en,occ] = r.text.split()
        en = float(en) - efermi
        arrband.append(en)
    arrband.sort()
    write.writerow(arrband)
    arrband = []
outPut.close()
print('finish')
