#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   test2.py    
@Time    :   2022/11/3 17:56  
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


def mn_test(*args, **kwargs):
    m_ = args[0]

    n_ = args[1]

    if m_**2 + n_**2 + m_*n_ == 3:
        return True
    return False


if __name__ == '__main__':
    # start here
    list_mn = [i for i in range(-10, 10)]
    for i in list_mn:
        for j in list_mn:
            if mn_test(i, j):
                print('i = %d, j = %d' % (i, j))