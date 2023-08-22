#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   边缘态SU4.py    
@Time    :   2022/4/22 16:33  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

from math import sqrt, pi

import numpy as np
from numpy import exp
from numpy.linalg import inv
from scipy.linalg import block_diag
from numpy.linalg import eigh
import matplotlib.pylab as plt
from numba import njit

t = -1
N = 64


# 生成块三对角矩阵函数
def tridiag(c, u, d, N):
    # c, u, d are center, upper and lower blocks, repeat N times
    cc = block_diag(*([c] * N))
    shift = c.shape[1]
    uu = block_diag(*([u] * N))
    uu = np.hstack((np.zeros((uu.shape[0], shift)), uu[:, :-shift]))
    dd = block_diag(*([d] * N))
    dd = np.hstack((dd[:, shift:], np.zeros((uu.shape[0], shift))))
    return cc + uu + dd


# 生成Hamiltonian动能部分
def H0_SU4(k):
    A = np.matrix([[sqrt(3), 0, 1, exp(-1j * k)],
                   [0, sqrt(3), 1, 1], [1, 1, sqrt(3), 0],
                   [exp(1j * k), 1, 0, sqrt(3)]])
    B = np.matrix([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [0, -1, 0, 0]])
    H = tridiag(A, B, B.H, N)
    H[0, 0] = 1000
    return H


def Zigzag_Graphene_H0(k):
    A = np.matrix([[0, t * (1 + exp(-1j * k))],
                   [t * (1 + exp(1j * k)), 0]])
    B = np.matrix([[0, 0],
                   [t, 0]])
    return tridiag(A, B, B.H, N)


def calculated_band(ks, Hk, m):
    nk = ks.size
    band = np.zeros((nk, m))
    for i in range(len(ks)):
        E, _ = eigh(Hk(ks[i]))
        band[i, :] = E
    return band


def plot_SU4_Band():
    nk = 64
    ks = np.linspace(0, 2 * pi, nk)
    band = calculated_band(ks, H0_SU4, 4 * N)
    plt.plot(band, color="gray")
    plt.plot(band[:, N - 1], color="red")
    plt.xticks(np.arange(0, nk, nk // 3), ['0', '2/3π', '4/3π', '2π'], fontsize=12, fontweight='bold')
    plt.yticks(np.arange(-0.5, 0.6, 0.5), fontsize=12)
    plt.ylim(-0.5, 0.5)
    plt.xlabel("k", fontsize=13)
    plt.ylabel("E", fontsize=13)  # fontweight='bold')
    plt.show()


if __name__ == '__main__':
    plot_SU4_Band()

    print('finish')