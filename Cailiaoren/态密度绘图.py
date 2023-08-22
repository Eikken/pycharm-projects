#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   态密度绘图.py    
@Time    :   2020/12/24 17:12  
@Tips    :    
'''

import random,sys,linecache,csv,math,time
start_time = time.time()
winSize = 0.0111
title = 'sdos.den'
energy = 'energy'

data = open('%s' % title)
outPut = open('data\DoS','w')
write = csv.writer(outPut)

for n,line in enumerate(data):
    m = 0.0
    dos = 0.0
    freq = float(line.split()[0])
    floor = freq - winSize
    roof = freq+winSize
    datagp = open('%s' % energy)
    for linegp in datagp:
        m = m + 1.0
        freqgp = float(linegp)
        freqgp = (freqgp+495.64757)/54
        if freqgp >= floor and freqgp <= roof:
            print(floor,roof)
            dos = dos + 1.0
    datagp.close()
    print('m:',m)
    dos = dos / m
    write.writerow([freq,dos])
    print('freq:',freq,'dos:',dos)
outPut.close()
print("use time :",str(time.time()-start_time))