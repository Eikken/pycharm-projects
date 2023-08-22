#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   提取后三列.py    
@Time    :   2023/5/20 16:28  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''


import numpy as np


if __name__ == '__main__':
    # start here
    fileName = r'data/2380.out'
    fileName2 = r'data/test.out'
    toFile = 'data/%s' % (fileName.split('/')[-1]+'.out')
    f = open(fileName)
    f.seek(0)
    f.readline()
    f.readline()

    f2 = open(fileName2)
    while True:
        fline = f2.readline()
        if 'Atom' in fline:
            break

    with open(toFile, 'w') as tof:
        while True:
            fline = f.readline()
            f2line = f2.readline()
            if fline == '':
                break
            line = fline.split()
            line2 = f2line.split()
            wline = "%s \t%s \t%s \t%s \t%s\n" % (line2[0], line2[1], line[-3], line[-2], line[-1])
            tof.write(wline)

    print('finished')
