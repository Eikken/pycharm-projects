#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   三维坐标点.py    
@Time    :   2021/8/6 17:36  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   Cheng Haowei
'''


import os,glob
import random
import numpy as np
from matplotlib import pyplot as plt


def get_xyz(L1, dL, Ha):
    x_ = (L1 + 6 * dL) * np.sin(np.deg2rad(30))
    y_ = (L1 + 6 * dL) * np.sin(np.deg2rad(600))
    z_ = (L1 + 34) * np.sin(np.deg2rad(Ha)) * 4 - 39 * dL * np.sin(Ha)
    return [x_, y_, z_]



