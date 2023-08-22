#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   三个文件提取数据.py    
@Time    :   2023/4/19 15:19  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   wsr
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def yaml2vesta(*args, **kwargs):
    param = str(kwargs['param'])
    # param = 'frequency:    0.9250759392'
    file1 = r'data/bulk28.vesta'
    file2 = r'data/mode_1519.86.vesta'
    file3 = r'data/freq30.txt'
    file4 = r'E:/桌面文件备份/27.8_freq/band.yaml'
    file5 = r'data/mode_%s.vesta' % param
    with open(file1) as f:
        lines1 = f.readlines()

    with open(file5, 'w') as f:

        for i in range(len(lines1)):
            if 'VECTR' in lines1[i]:
                break
            f.write(lines1[i])

    allss = []
    with open(file3) as f:
        f.seek(0)
        while True:
            lines3 = f.readline()
            if param in lines3:
                break

        while True:
            lines3 = f.readline()
            if lines3 == '':
                break
            if '# atom' in lines3:
                atomNum = lines3.split()[-1]
                real = [atomNum]
                for j in range(3):
                    thisline = f.readline().split()[2]
                    real.append(thisline[:len(thisline) - 1])
                ss = '   %s  %s %s %s\n' % (real[0], real[1], real[2], real[3])
                ss2 = '    %s  0   0    0    0\n 0 0 0 0 0\n' % real[0]
                ss += ss2
                allss.append(ss)

    with open(file5, 'a') as f:
        f.write('VECTR\n')
        for i in allss:
            f.write(i)
        f.write(' 0 0 0 0 0\nVECTT\n')
        for i in range(len(allss)):
            ss = '''   %d 0.500 255   0   0 1
     0 0 0 0 0\n''' % (i + 1)
            f.write(ss)
    print('finished')


yaml2vesta(param='0.9250759392')

