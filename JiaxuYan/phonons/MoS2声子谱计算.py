#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   MoS2声子谱计算.py    
@Time    :   2023/2/20 15:13  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
    # momo = l_ ** 2 * Vsig3 + (1 - l_ ** 2) * Vpi3
    # xuxu = m_ ** 2 * Vsig3 + (1 - m_ ** 2) * Vpi3
    # xdxd = n_ ** 2 * Vsig3 + (1 - n_ ** 2) * Vpi3
    # moxu = -l_ * m_ * (Vsig1 - Vpi1)
    # moxd = -l_ * n_ * (Vsig1 - Vpi1)
    # xuxd = -m_ * n_ * (Vsig2 - Vpi2)

'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

default_params = {
    # from https://iopscience.iop.org/article/10.1088/1361-648X/ac5539#cmac5539f2
    # ->     [[  Mo-Xu(d) ], [   Xu-Xd  ], [Mo^2],[Xu(d)^2], [  Mo-Xu(d) ],  [   Xu-Xd   ], [   Mo-Xu(d)  ]]
    # ->       Vppσ1, Vppπ1,  Vppσ2,  Vppπ2,  Vppσ3,  Vppπ3,  Vppσ4, Vppπ4,  Vppσ5,  Vppπ5,  Vppσ6,  Vppπ6,
    "MoS2": [24.358, 9.442, 51.903, 19.306, 5.6020, 0.1700, -7.582, 2.914, -4.579, 0.6300, 1.4930, -0.254],
    "MoSe2": [19.656, 3.478, 1.9530, -0.779, 2.4030, -0.770, 1.5510, 0.612, 1.8540, -0.770, -1.107, 0.652],
    "MoTe2": [14.562, 1.492, 2.0590, -0.233, 1.2340, -0.099, -0.079, 0.105, -0.676, -0.057, -0.326, 0.228],
}


def R_phi(theta_):
    return np.array([[np.cos(theta_), np.sin(theta_), 0],
                     [-np.sin(theta_), np.cos(theta_), 0],
                     [0, 0, 1]])


def K_matrix(theta_, k_):
    U = np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])
    return np.linalg.inv(U) @ k_ @ U  # K矩阵左乘右乘rotate矩阵


def rotate(theta_, r_):
    RA = []
    RB = []
    RC = []
    Su = np.array([[d]])
    Sd = np.array([[-d]])
    Um = np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])
    U = np.array([[np.cos(np.pi), np.sin(np.pi), 0], [-np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1]])
    for im in range(int(2 * np.pi / theta_)):
        if np.size(r_) == 3:
            r_ = np.matmul(np.linalg.inv(Um), r_) * np.sin(theta)  # M atom d = 0
            ra_ = r_
            rb_ = np.row_stack((-1*(r_[:2]), Su))  # S atom up d = d
            rc_ = np.row_stack((-1*(r_[:2]), Sd))  # S atom down d = -d
        else:
            r_ = np.linalg.inv(Um) @ r_ @ Um  # @ 表示矩阵乘法(不常用)  # M atom d = 0
            ra_ = r_
            rb_ = np.linalg.inv(U) @ np.row_stack((-1 * (r_[:2]), Su)) @ U  # S atom up d = d
            rc_ = np.linalg.inv(U) @ np.row_stack((-1 * (r_[:2]), Sd)) @ U  # S atom down d = -d
        RA.append(ra_)
        RB.append(rb_)
        RC.append(rc_)
    return RA, RB, RC


def kij(l_, m_, n_):
    # l_, m_, n_ : x, y, z方向的cosine值
    momo = l_ ** 2 * Vsig3 + (1 - l_ ** 2) * Vpi3
    xuxu = m_ ** 2 * Vsig3 + (1 - m_ ** 2) * Vpi3
    xdxd = n_ ** 2 * Vsig3 + (1 - n_ ** 2) * Vpi3
    moxu = -l_ * m_ * (Vsig1 - Vpi1) - l_ * m_ * (Vsig4 - Vpi4) - l_ * m_ * (Vsig6 - Vpi6)
    moxd = -l_ * n_ * (Vsig1 - Vpi1) - l_ * n_ * (Vsig4 - Vpi4) - l_ * n_ * (Vsig6 - Vpi6)
    xuxd = -m_ * n_ * (Vsig2 - Vpi2) - m_ * n_ * (Vsig5 - Vpi5)


def Dym_(k_):
    kmomo = np.zeros((3, 3), dtype=complex)
    kxuxu = np.zeros((3, 3), dtype=complex)
    kxdxd = np.zeros((3, 3), dtype=complex)
    kmoxu = np.zeros((3, 3), dtype=complex)
    kmoxd = np.zeros((3, 3), dtype=complex)
    kxuxd = np.zeros((3, 3), dtype=complex)

    for i in range(3):  # xyz or lmn
        pass
    # layer 2

    # layer 3

    # layer 4

    Kij = np.zeros((9, 9), dtype=complex)

    return Kij


if __name__ == '__main__':
    # start here
    time1 = time.time()
    # a = 1.42e-10
    # m = 1.99e-26
    a = 3.16  # A
    d = 2.40  # A
    theta = np.deg2rad(40.6)  # 参考文献

    superatoms = 14
    n = 100

    rt3 = 3 ** 0.5
    # 以A为圆心周围原子坐标
    pi = np.pi
    FA, FB, FC = rotate(2 / 3 * pi, np.array([[1], [0], [0]]) * a)  # 第一近邻
    SA, SB, SC = rotate(1 / 3 * pi, np.array([[3 / 2], [rt3 / 2], [0]]) * a)  # 第二近邻
    TA, TB, TC = rotate(2 / 3 * pi, np.array([[1], [rt3], [0]]) * a)  # 第三近邻
    LA1, LB1, LC1 = rotate(2 / 3 * pi, np.array([[2.5], [rt3 / 2], [0]]) * a)  # 第四近邻，两种角度
    LA2, LB2, LC2 = rotate(2 / 3 * pi, np.array([[2.5], [-rt3 / 2], [0]]) * a)
    LA = LA1 + LA2  # 第四近邻两种情况进行合并，可省略
    LB = LB1 + LB2
    LC = LC1 + LC2

    params = default_params['MoS2']
    [Vsig1, Vpi1, Vsig2, Vpi2, Vsig3, Vpi3, Vsig4, Vpi4, Vsig5, Vpi5, Vsig6, Vpi6] = [ii for ii in params]


