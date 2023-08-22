#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于画圈求圈层.py    
@Time    :   2023/4/10 18:43  
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


def get_sphere(*args, **kwargs):
    sp_ = args[0]
    spDict = {}
    # sp = 0
    sp0 = [[0, 0, 0]]
    spDict['sp0'] = sp0
    # sp = 1 >> 3*2
    sp1 = [[1, 0, 0], [1, 1, 0], [0, 1, 0]]
    sp1 = sp1 + [list(-np.array(sp1[i])) for i in range(len(sp1))]
    spDict['sp1'] = sp1
    # sp = 2 >> 3*2
    sp2 = [[2, 1, 0], [1, 2, 0], [-1, 1, 0]]
    sp2 = sp2 + [list(-np.array(sp2[i])) for i in range(len(sp2))]
    spDict['sp2'] = sp2
    # sp = 3 >> 3*2
    sp3 = [[2, 0, 0], [2, 2, 0], [0, 2, 0]]
    sp3 = sp3 + [list(-np.array(sp3[i])) for i in range(len(sp3))]
    spDict['sp3'] = sp3
    # sp = 4 >> 6*2
    sp4 = [[3, 1, 0], [3, 2, 0], [2, 3, 0], [1, 3, 0], [-1, 2, 0], [-2, 1, 0]]
    sp4 = sp4 + [list(-np.array(sp4[i])) for i in range(len(sp4))]
    spDict['sp4'] = sp4
    # sp = 5 >> 3*2
    sp5 = [[3, 0, 0], [3, 3, 0], [0, 3, 0]]
    sp5 = sp5 + [list(-np.array(sp5[i])) for i in range(len(sp5))]
    spDict['sp5'] = sp5
    # sp = 6 >> 3*2
    sp6 = [[4, 2, 0], [2, 4, 0], [-2, 2, 0]]
    sp6 = sp6 + [list(-np.array(sp6[i])) for i in range(len(sp6))]
    spDict['sp6'] = sp6
    # sp = 7 >> 3*2
    sp7 = [[4, 1, 0], [4, 3, 0], [3, 4, 0], [1, 4, 0], [-1, 3, 0], [-3, 1, 0]]
    sp7 = sp7 + [list(-np.array(sp7[i])) for i in range(len(sp7))]
    spDict['sp7'] = sp7
    # sp = 8 >> 3*2
    sp8 = [[4, 0, 0], [4, 4, 0], [0, 4, 0]]
    sp8 = sp8 + [list(-np.array(sp8[i])) for i in range(len(sp8))]
    spDict['sp8'] = sp8
    sp9 = [[5, 2, 0], [5, 3, 0], [2, 5, 0], [3, 5, 0], [-2, 3, 0], [-3, 2, 0]]
    sp9 = sp9 + [list(-np.array(sp9[i])) for i in range(len(sp9))]
    spDict['sp9'] = sp9
    sp10 = [[5, 1, 0], [5, 4, 0], [1, 5, 0], [4, 5, 0], [-1, 4, 0], [-4, 1, 0]]
    sp10 = sp10 + [list(-np.array(sp10[i])) for i in range(len(sp10))]
    spDict['sp10'] = sp10
    tmpl = []
    if sp_ == 0:
        return sp0

    for i in range(sp_):
        tmpl += [j for j in spDict['sp%d' % i]]
    return tmpl


if __name__ == '__main__':
    # start here
    spList = get_sphere(0)
    print(spList)