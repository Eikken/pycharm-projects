#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   D3h轨道对称性测试.py    
@Time    :   2023/3/28 10:02  
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
import sympy

params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22
    "MoS2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057],
    "WS2":   [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061],
    "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290,  0.130],
    "WSe2":  [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329,  0.034],
    "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239,  0.252],
    "WTe2":  [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270,  0.190],
}


def Ur(*args, **kwargs):
    theta_ = np.deg2rad(args[0])
    pi = sympy.symbols('π')
    return np.array([[np.cos(theta_), np.sin(theta_), 0],
                    [-np.sin(theta_), np.cos(theta_), 0],
                    [             0,              0,  1]])


if __name__ == '__main__':
    # start here
    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params.copy()['MoS2']
    rt3 = 3 ** 0.5
    # rt3 = sympy.sqrt(3)
    # sympy.pprint()
    h1 = [[t0, -t1, t2],
          [t1, t11, -t12],
          [t2, t12, t22]]

    h2 = [[t0, 1 / 2 * t1 + rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [-1 / 2 * t1 + rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t11 - t22) - t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t11 - t22) + t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h3 = [[t0, -1 / 2 * t1 - rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [1 / 2 * t1 - rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t22 - t11) + t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t22 - t11) - t12, 3 / 4 * t11 + 1 / 4 * t22]]

    # sympy.pprint(sympy.Matrix(h1))
    print('U 120°')
    # sympy.pprint(sympy.Matrix(h2))
    Uh3 = sympy.Matrix(np.linalg.inv(Ur(120)) @ h1 @ Ur(120))
    sympy.pprint(sympy.Matrix(Uh3))
    print('h2')
    sympy.pprint(sympy.Matrix(h2))
    print('h3')
    sympy.pprint(sympy.Matrix(h3))

