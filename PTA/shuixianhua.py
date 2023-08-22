#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   shuixianhua.py    
@Time    :   2022/10/22 15:18  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time


# import math
# import matplotlib.pyplot as plt
# import pybinding as pb
# import pandas as pd
# import numpy as np


def your_func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    time1 = time.time()
    # write here
    for i in range(300, 380+1):
        a = i%10
        b = i//10%10
        c = i//100%10
        if a**3 + b**3+c**3 == i:
            print(i)


    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))