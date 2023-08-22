#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   onsiteTest.py    
@Time    :   2022/9/7 10:55  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   对于MoS2，没有合适的fit参数，先测试Graphene吧。
            对于VppSigma, VppDelta, VppPi 不同组合实现夹角、位置的hopping参数设置

'''

import time

import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np


def your_func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    time1 = time.time()
    # write here
    a_ = 3.19032
    print(a_*3**0.5/2)
    print(a_ / 2)

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))