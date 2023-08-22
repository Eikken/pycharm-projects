#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   分子动力学数据模拟计算.py
@Time    :   2020/12/18 11:54  
'''

import random,sys,linecache,csv,math,time
import shutil, os
import xml.etree.ElementTree as ET
import numpy as np
import numpy.linalg as la

start_time = time.time()

sFrame = 1000 #
numFrame = 5 # 几个皮秒等
stepSize = 10
totalAtom = 324

def py_ang( v1, v2):
    # 向量X乘 求两个向量的夹角
    cosang = np.dot(v1,v2)
    sinang = la.norm(np.cross(v1.v2))
    rang = np.arctan2(sinang,cosang)
    angle = rang*180/np.pi
    return angle

data = open('data/XDATCAR')
lines = data.readlines()

outcella = open('data\\a.txt','w')
outcellb = open('data\\b.txt','w')
outcellc = open('data\\c.txt','w')
outalpha = open('data\\alpha.txt','w')
outbeta = open('data\\beta.txt','w')
outgamma = open('data\\gamma.txt','w')
outvol = open('data\\volume.txt','w')

aveax,aveay,aveaz = [],[],[]
avebx,aveby,avebz = [],[],[]
avecx,avecy,avecz = [],[],[]
xns = 0
for n,line in enumerate(lines):
    if 'configuration=' in line.split():
        ns = float(line.split()[2])
        if ns >= sFrame:
            [v1x, v1y, v1z] = linecache.getline('data/XDATCAR', n - 4).split()
            va = [float(v1x), float(v1y), float(v1z)]
            aveax.append(va[0])
            aveay.append(va[1])
            aveaz.append(va[2])
            [v2x, v2y, v2z] = linecache.getline('data/XDATCAR', n - 3).split()
            vb = [float(v2x), float(v2y), float(v2z)]
            avebx.append(vb[0])
            aveby.append(vb[1])
            avebz.append(vb[2])
            [v3x, v3y, v3z] = linecache.getline('data/XDATCAR', n - 2).split()
            vc = [float(v3x), float(v3y), float(v3z)]
            avecx.append(vc[0])
            avecy.append(vc[1])
            avecz.append(vc[2])

            cella = la.norm(va)
            cellb = la.norm(vb)
            cellc = la.norm(vc)
            alpha = py_ang(vb, vc)
            beta = py_ang(va, vc)
            gamma = py_ang(va, vb)
            crossv = np.cross(va, vb)
            volume = np.dot(vc,crossv)

            xns = (ns - sFrame)/stepSize
            outcella.write(str(xns) + ' ' + str(cella) + '\n')
            outcellb.write(str(xns) + ' ' + str(cellb) + '\n')
            outcellc.write(str(xns) + ' ' + str(cellc) + '\n')
            outalpha.write(str(xns) + ' ' + str(alpha) + '\n')
            outbeta.write(str(xns) + ' ' + str(beta) + '\n')
            outgamma.write(str(xns) + ' ' + str(gamma) + '\n')
            outvol.write(str(xns) + ' ' + str(volume) + '\n')
outcella.close()
outcellb.close()
outcellc.close()
outalpha.close()
outbeta.close()
outgamma.close()
outvol.close()
avax = sum(aveax) / (xns + 1)
avay = sum(aveay) / (xns + 1)
avaz = sum(aveaz) / (xns + 1)
avbx = sum(avebx) / (xns + 1)
avby = sum(aveby) / (xns + 1)
avbz = sum(avebz) / (xns + 1)
avcx = sum(avecx) / (xns + 1)
avcy = sum(avecy) / (xns + 1)
avcz = sum(avecz) / (xns + 1)

output = open('aveconfig','w')
output.write(linecache.getline('data/XDATCAR', 1))
output.write(linecache.getline('data/XDATCAR', 2))
output.write(str(avax) + ' ' + str(avay) + ' ' + str(avaz) + '\n')
output.write(str(avbx) + ' ' + str(avby) + ' ' + str(avbz) + '\n')
output.write(str(avcx) + ' ' + str(avcy) + ' ' + str(avcz) + '\n')
output.write(linecache.getline('data/XDATCAR', 6))
output.write(linecache.getline('data/XDATCAR', 7))
output.write(linecache.getline('data/XDATCAR', 8))
for atomNum in range(totalAtom):
    xl,yl,zl = [],[],[]
    data = open('data/XDATCAR')
    lines = data.readlines()
    for n, line in enumerate(lines):
        if 'configuration=' in line.split():
            ns = float(line.split()[2])
            if ns == sFrame:
                xs = float(linecache.getline('data/XDATCAR', n + 1 + atomNum + 1).split()[0])
                ys = float(linecache.getline('data/XDATCAR', n + 1 + atomNum + 1).split()[1])
                zs = float(linecache.getline('data/XDATCAR', n + 1 + atomNum + 1).split()[2])
            if ns >= sFrame:
                num = n+1+atomNum+1
                x = float(linecache.getline('data/XDATCAR', num).split()[0])
                y = float(linecache.getline('data/XDATCAR', num).split()[1])
                z = float(linecache.getline('data/XDATCAR', num).split()[2])
                if np.abs(x - xs) > 0.5:
                    print('大于0.5',x)
                    x=x-1.0*np.sign(x-xs)
                if np.abs(y - ys) > 0.5:
                    y = y - 1.0 * np.sign(y - ys)
                if np.abs(z - zs) > 0.5:
                    z = z - 1.0 * np.sign(z - zs)
                xl.append(x)
                yl.append(y)
                zl.append(z)
    avcdx = sum(xl) / (xns + 1)
    avcdy = sum(yl) / (xns + 1)
    avcdz = sum(zl) / (xns + 1)
    output.write(str(avcdx) + ' ' + str(avcdy) + ' ' + str(avcdz) +'\n')
    data.close()
output.close()
print('use time:',time.time()-start_time,'s')