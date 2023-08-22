#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   删减64.py    
@Time    :   2023/4/13 18:49  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get64(*args):
    data = open(args[0])
    # lines = data.readlines()
    data.seek(0)
    data.readline()
    first_line = ' 38 38\n'
    flag = 64 * 103 * 4 + 1
    while True:
        flag -= 1
        if flag == 0:
            break
        line = data.readline()  # 64 64

    flag2 = 39

    with open(args[1], 'w') as f:
        f.write(first_line)
        while True:
            for i in range(0, 64):
                for j in range(4):
                    data.readline()

            for i in range(64, 103):
                [x, y] = data.readline().split()
                f.write('%d %d' % (int(x) - 64, int(y) - 64))
                for j in range(3):
                    f.write(data.readline())

            flag2 -= 1
            if flag2 == 0:
                break


if __name__ == '__main__':
    # start here

    file_name = 'data/FORCE_CONSTANTS_c'
    to_file_name = 'data/FORCE_CONSTANTS_cdrop'
    # get300(file_name, to_file_name, 103)  # 663 1026 1752
    get64(file_name, to_file_name)
    print('finish')