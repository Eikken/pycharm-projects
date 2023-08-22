#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   处理曲面S.py    
@Time    :   2022/12/5 15:19  
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


def func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    # start here
    data = open('data/POSCAR221205')
    lines = data.readlines()
    index = lines.index('Direct\n')

    data_list = []
    atom_list = []
    title_list = []
    for i in range(len(lines)):
        if i <= index:
            title_list.append(lines[i])
        else:
            line = lines[i]
            if len(line.split()) >= 3:
                data_list.append([float(line.split()[j+1]) for j in range(3)])
                atom_list.append(line.split()[0])
    data_list = np.array(data_list)
    data_1 = data_list[np.where(data_list[:, 2] < 0.1)]
    # 0.0600379999999999; 0.0535739999999999
    max_value, min_value = data_1[:, 2].max(), data_1[:, 2].min()
    sub_value = max_value - (max_value - min_value) / 2
    # for i in range(len(data_list)): if data_list[i][2] < sub_value: # print('O', data_list[i]) print('%s   %.16f
    # %.16f   %.16f   \n' % ('O', data_list[i][0], data_list[i][1], data_list[i][2])) else: # print(atom_list[i],
    # data_list[i]) print('%s   %.16f   %.16f   %.16f   \n' % (atom_list[i], data_list[i][0], data_list[i][1],
    # data_list[i][2]))
    sat = []
    with open('data/poscar221205o', 'w') as writer:
        for t in title_list:
            writer.write(t)
        for i in range(len(data_list)):
            if data_list[i][2] < sub_value:
                # print('O', data_list[i])
                sat.append('%s   %.16f   %.16f   %.16f   \n' % ('O', data_list[i][0], data_list[i][1], data_list[i][2]))
            else:
                # print(atom_list[i], data_list[i])
                writer.write('%s   %.16f   %.16f   %.16f   \n' % (atom_list[i], data_list[i][0], data_list[i][1], data_list[i][2]))
        for i in sat:
            writer.write(i)

        writer.write('\n')

    # data.seek(0)  # just in case
    # while True:
    #     line = data.readline()
    #     if not line:
    #         break
    #     if "Direct" in line:
    #         print(data.readline())
    #         print(data.readline())
    #         break
    #     #
    print('finish')