#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   lwx220714.py
@Time    :   2022/7/14 16:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np


def func(y_):
    return np.exp(np.log(y_/1499)/0.28146)


if __name__ == '__main__':
    y = 2196
    # print(func(y))
    # print(3.88**0.28146*1499)
    l = 3.19
    h = 1.8
    tanhl = h / l
    print(np.rad2deg(np.arctan(tanhl)))