#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   DigitMode.py    
@Time    :   2022/9/9 9:58  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time


# import math
# import matplotlib.pyplot as plt
# import pybinding as pb
# import pandas as pd
import numpy as np


def mxValue(x_):
    listNum = list(map(int, str(x_)))
    counts = np.bincount(listNum)
    # print(counts)
    return np.argmax(counts)


if __name__ == '__main__':
    time1 = time.time()
    # write here
    mod_ = np.power(10, 9) + 7
    numList = [9, 99, 999, 9999]
    for n in numList:

        print(np.sum([mxValue(i) for i in range(n+1)]))

    # print(mxValue(99))
    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))