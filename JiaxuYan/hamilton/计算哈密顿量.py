#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   计算哈密顿量.py    
@Time    :   2021/12/24 16:26  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import copy
import functools
import itertools
import sys

import networkx as nx
from scipy.spatial import cKDTree
from JiaxuYan.绘制AB_AA_twist重叠度对比 import getDict, dropMethod
import math
import numpy as np
import xlwt
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd


# N = 10

def hamiltonian(k1, k2, M, t1, a=1 / np.sqrt(3)):
    # initial 0 matrix of h0, h1
    h0 = np.zeros((2, 2), dtype=complex)
    h1 = np.zeros((2, 2), dtype=complex)
    # H0 = np.zeros((2, 2), dtype=complex)
    # H1 = np.zeros((2, 2), dtype=complex)
    # Mass term, open band gap
    h0[0, 0], h0[1, 1] = M, -M
    # nearest neighbor
    # 1j/2*k2*a = τ; τ is the period of A-B distance
    h1[1, 0] = t1 * (np.exp(1j * k2 * a) +
                     np.exp(1j * np.sqrt(3) / 2 * k1 * a - 1j / 2 * k2 * a) +
                     np.exp(-1j * np.sqrt(3) / 2 * k1 * a - 1j / 2 * k2 * a))
    # conjunction matrix
    h1[0, 1] = h1[1, 0].conj()
    # N = 10
    # 第一个关于A的方程里，
    # t3 项表示与同一个原胞内B之间跃迁，
    # t2 项表示与左边相邻原胞内B的跃迁（所以要乘上Bloch因子，表示向左平移一个原胞里B的波函数），
    # t1 项表示与下面一层的B的跃迁。
    # H0[1, 0] = t1 + t1*np.exp(-2*np.pi*i*k1)
    # H= np.kron(np.diag(np.ones(N)),H0) + \
    #    np.kron(np.diag(np.ones(N-1),1),H1) + \
    #    np.kron(np.diag(np.ones(N-1),-1),np.conjugate(np.transpose(H1)))
    matrix = h0 + h1
    return matrix


if __name__ == '__main__':
    constant_a = 1.42
    hamiltonian0 = functools.partial(hamiltonian, M=0, t1=1, a=constant_a)  # 固定参数的偏函数
    xx = -4 * 3 ** (-np.pi / 2 / 9 * constant_a)
    kx = np.linspace(xx, -xx, 100)
    ky = np.linspace(xx, -xx, 100)
    dim = hamiltonian0(0, 0).shape[0]
    dim1 = kx.shape[0]
    dim2 = ky.shape[0]
    eigenvalue_k = np.zeros((dim2, dim1, dim))  # 本征值
    i0 = 0
    for kx0 in kx:
        j0 = 0
        for ky0 in ky:
            matrix0 = hamiltonian0(kx0, ky0)
            eigenvalue, eigenVector = np.linalg.eig(matrix0)  # return 特征值, 特征向量
            eigenvalue_k[j0, i0, :] = np.sort(np.real(eigenvalue[:]))
            j0 += 1
        i0 += 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    kx, ky = np.meshgrid(kx, ky)
    # print(eigenvalue_k[:, :, 1])
    ax.plot_surface(kx, ky, eigenvalue_k[:, :, 0], rcount=200, ccount=200, cmap='rainbow', linewidth=1,
                    antialiased=False)
    ax.plot_surface(kx, ky, eigenvalue_k[:, :, 1], rcount=200, ccount=200, cmap='rainbow_r', linewidth=1,
                    antialiased=False)
    # for d in range(dim):

        # ax.plot_surface(kx, ky, eigenvalue_k[:, :, 1], rcount=200, ccount=200, cmap='rainbow', linewidth=1,
        #             antialiased=False)
    # ax.contour(kx, ky, eigenvalue_k[:, :, 1], zdir='z', offset=0, cmap=plt.get_cmap('rainbow'))
    plt.show()