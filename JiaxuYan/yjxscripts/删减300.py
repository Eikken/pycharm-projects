#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   删减300.py    
@Time    :   2022/12/13 12:33  
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


def get300(*args, **kwargs):
    data = open(args[0])
    num = args[2]
    # lines = data.readlines()
    data.seek(0)
    first_line = data.readline().replace(str(num), '300')
    flag = 300
    with open(args[1], 'a') as f:
        f.write(first_line)
        while True:
            for i in range(1, 301):
                for j in range(4):
                    f.write(data.readline())
            for i in range(301, num+1):
                for j in range(4):
                    data.readline()
            print(str(300 - flag) + '\n' if flag % 50 == 0 else '.', end='')

            flag -= 1
            if flag == 0:
                break


def get3004(*args, **kwargs):
    data = open(args[0])
    # lines = data.readlines()
    data.seek(0)
    data.readline()
    first_line = ' 300 1200\n'
    flag = 300
    # [i*4+1 for i in range(5)]
    with open(args[1], 'a') as f:
        f.write(first_line)
        while True:
            for i in range(1, 300 * 4 + 1):
                for j in range(4):
                    f.write(data.readline())
            for i in range(300 * 4 + 1, 663 * 4 + 1):
                for j in range(4):
                    data.readline()
            flag -= 1
            if flag == 0:
                break


def get1200(*args, **kwargs):
    data = open(args[0])
    # lines = data.readlines()
    data.seek(0)
    first_line = data.readline().replace('2652', '1200')
    flag = 1200
    with open(args[1], 'a') as f:
        f.write(first_line)
        while True:
            for i in range(1, 300*4+1):
                for j in range(4):
                    f.write(data.readline())
            for i in range(300*4+1, 663*4+1):
                for j in range(4):
                    data.readline()
            print(str(1200 - flag) + '\n' if flag % 50 == 0 else '.', end='')

            flag -= 1
            if flag == 0:
                break


def get64(*args):
    data = open(args[0])
    # lines = data.readlines()
    data.seek(0)
    data.readline()
    first_line = ' 38 38\n'
    flag = 64*103*4 + 1
    while True:
        flag -= 1
        if flag == 0:
            break
        line = data.readline()  # 64 64

    flag2 = 39
    x = 1
    y = 1
    with open(args[1], 'a') as f:
        f.write(first_line)
        while True:
            for i in range(0, 64):
                for j in range(4):
                    data.readline()

            for i in range(64, 103):

                for j in range(3):
                    f.write(data.readline())

            flag2 -= 1
            if flag2 == 0:
                break
    # with open(args[1], 'a') as f:
    #     f.write(first_line)
    #     f.write(line)
    #     while True:
    #         for i in range(1, 39 * 4 + 1):
    #             for j in range(4):
    #                 f.write(data.readline())
    #         for i in range(0, 64):
    #             print(data.readline())
    #             # for j in range(4):
    #                 # data.readline()
    #         print(str(1200 - flag) + '\n' if flag % 50 == 0 else '.', end='')
    #
    #         flag -= 1
    #         if flag == 0:
    #             break


if __name__ == '__main__':
    # start here

    file_name = 'data/FORCE_CONSTANTS_c'
    to_file_name = 'data/FORCE_CONSTANTS_cdrop'
    # get300(file_name, to_file_name, 103)  # 663 1026 1752
    get64(file_name, to_file_name)
    print('finish')
