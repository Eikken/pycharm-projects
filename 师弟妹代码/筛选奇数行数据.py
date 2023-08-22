#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   筛选奇数行数据.py    
@Time    :   2023/4/12 18:29  
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
    file1 = r'data/赵寻.xlsx'
    df = pd.DataFrame(pd.read_excel(file1))[['c', 'd']].values
    data = []
    for i in range(len(df)):
        if i % 2 == 0:
            data.append(df[i])
    pd.DataFrame(data).to_excel('data/偶数cd行.xls')