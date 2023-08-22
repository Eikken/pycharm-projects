#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   ase矩阵.py    
@Time    :   2023/4/12 14:29  
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
    spm = np.diag([7**0.5, 7**0.5, 1])
    dignals = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ])
    dotp = np.dot(dignals, spm)
    mmin = np.min(dotp, axis=0)
    mmax = np.max(dotp, axis=0) + 1

    ar = np.arange(mmin[0], mmax[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mmin[0], mmax[0])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mmin[0], mmax[0])[:, None] * np.array([0, 0, 1])[None, :]

    br = br[None, :, None]
    print(br)