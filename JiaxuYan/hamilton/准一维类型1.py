#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   准一维类型1.py    
@Time    :   2022/6/23 10:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   write by celeste about 1D transform graphene
'''

import numpy as np
import functools as ft
import matplotlib.pyplot as plt
import time


def OneD8AtomFunc(kx_):
    t_ = -1
    gamma_v = 0.1 * t_  # 质量项，打开带隙，相当于M
    gamma_so = 0.06*t_  # 次近邻自旋轨道耦合，相当于t2
    gamma_r = 0.05*t_  # Rashba类型自旋轨道耦合项
    N = 20

    sx_values = np.array([1, 0])
    sxn_values = np.max(sx_values) + 1  # 网上学的 one-hot 矩阵形式
    s0 = np.eye(4, dtype=int)
    sx = np.eye(sxn_values)[sx_values]
    sy = np.asmatrix([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    v1 = np.conj([[3 ** 0.5, 0]]).T
    v2 = np.conj([[-3 ** 0.5 / 2, 3 / 2]]).T
    v3 = np.conj([[-3 ** 0.5 / 2, -3 / 2]]).T

    # 把state换成2一样可以说清楚
    # k_ = np.mat([kx_, ky_])

    H0 = np.zeros((8, 8), dtype=complex)
    t1_ = -t_
    H0 = H0 + np.kron(gamma_v*s0, sz) + t1_ * np.eye(8, k=1) + t1_ * np.eye(8, k=-1)  # Kronecker product of two arrays.
    t2_down = np.array([-1, -1, 1, 1, -1, -1, 0, 0])  # k=-2 k取正负与对角线如何生成有重要关系
    t2_up = np.array([0, 0, 1, 1, -1, -1, 1, 1])  # 最终结果是将主对角线向上或者向下平移 ，注意生成k array

    T0 = H0 + gamma_so * np.exp(1j*t2_down) * np.eye(8, k=-2) + gamma_so * np.exp(1j*t2_up) * np.eye(8, k=2)
    # print(H0)

    TX = np.zeros((8, 8), dtype=complex)
    TX_diagArr = np.array([np.kron(np.eye(4), sz)[i][i] for i in range(8)])
    TX = TX + gamma_so * np.exp(1j*TX_diagArr)*np.eye(8, k=0)
    TX_up = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    TX_down = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    TX = TX + np.eye(8, k=1)*TX_up*t_ + np.eye(8, k=-1) * TX_down * t_
    TX_2up = np.array([0, 0, 1, 0, 0, 1, -1, 0])
    TX_2down = np.array([0, 1, -1, 0, 0, 1, 0, 0])
    TX = TX + gamma_so * np.exp(1j * TX_2down) * np.eye(8, k=-2) + gamma_so * np.exp(1j * TX_2up) * np.eye(8, k=2)

    TY = np.zeros((8, 8), dtype=complex)
    TY[7][0] = t_
    TY[6][0] = gamma_so * np.exp(1j * t_)
    TY[7][1] = gamma_so * np.exp(1j * t_)

    Tyx = np.zeros((8, 8), dtype=complex)
    Tyx[7][1] = gamma_so * np.exp(1j * -t_)
    Txy = np.zeros((8, 8), dtype=complex)
    Txy[0][6] = gamma_so * np.exp(1j * -t_)

    H00 = np.kron(np.eye(N, k=1), TY)
    H00 = np.kron(np.eye(N), T0) + H00 + np.conj(H00).T
    H01 = np.kron(np.eye(N), TX) + np.kron(np.eye(N, k=1), Tyx) + np.kron(np.eye(N, k=-1), Txy)
    H = H00 + H01 * np.exp(1j * kx_ * np.pi) + np.conj(H01).T * np.exp(-1j * kx_ * np.pi)
    return H


def OneD8AtomPlot(x_, OneD8Ham):
    dim = OneD8Ham(0).shape[0]
    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    eigen_k = np.zeros((dim1, dim))
    i0 = 0
    for x0 in x_:
        matrix0 = OneD8Ham(x0)
        eigenValue, eigenVector = np.linalg.eig(matrix0)
        eigen_k[i0, :] = np.sort(np.real(eigenValue[:]))
        i0 += 1
    # plt.figure(figsize=(7, 4))
    plt.plot(x_, eigen_k, linewidth=0.5, color='black')
    # plt.ylim((-1, 1))
    plt.title('1D 8 Atoms')
    plt.show()


if __name__ == '__main__':
    time1 = time.time()
    num = 100
    kx = np.linspace(-1, 1, num)
    ky = np.linspace(-1, 1, num)
    # OneD8AtomFunc(0)
    ham0 = ft.partial(OneD8AtomFunc)

    OneD8AtomPlot(kx, ham0)
    time2 = time.time()

    print('finish', end=' ')
    print('use time %d s' % (time2-time1))