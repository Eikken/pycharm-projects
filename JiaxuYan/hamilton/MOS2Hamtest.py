#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   MOS2Hamtest.py    
@Time    :   2022/7/26 13:37  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   S atoms are 1.56A above and below the Mo plane.
             Angle between Mo-S is 40.6°
'''

import time

import numpy as np
import functools as ft
import matplotlib.pyplot as plt


if __name__ == '__main__':
    time1 = time.time()
    ccDis = 1.42  # Å, nearest c-c bond length
    vf = 5.944  # eV·Å Fermi velocity
    phi = 2 * np.pi / 3
    agl = 1.05
    theta = np.deg2rad(agl)
    w1 = 0.110
    A_mos2 = 3.16
    R1 = np.array([1, 0, 0]) * A_mos2
    R2 = np.array([1/2, 3**0.5/2, 0]) * A_mos2
    # w1 = 0
