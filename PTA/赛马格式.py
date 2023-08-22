#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   赛马格式.py    
@Time    :   2022/10/22 14:28  
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

    # 方式1
    import sys

    lines = sys.stdin.readlines()

    allLines = [item.split() for item in lines if item != '\n']

    # 方式2
    dataSet = []
    while True:
        line = input()
        if line == '':
            break
        dataSet.append(list(map(int, line.split())))


