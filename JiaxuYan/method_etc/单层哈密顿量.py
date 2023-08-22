#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   单层哈密顿量.py    
@Time    :   2021/5/7 12:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath
import functools


def hamiltonian(k1, k2, M, t1, a=1 / sqrt(3)):
    # 初始化0矩阵
    h0 = np.zeros((2, 2)) * (1 + 0j)
    h1 = np.zeros((2, 2)) * (1 + 0j)
    h2 = np.zeros((2, 2)) * (1 + 0j)
    #  质量项，用于打开带隙
    h0[0, 0] = M
    h0[1, 1] = -M

    # 最近邻项
    h1[1, 0] = t1 * (cmath.exp(1j * k2 * a) + cmath.exp(1j * sqrt(3) / 2 * k1 * a - 1j / 2 * k2 * a)
                     + cmath.exp(-1j * sqrt(3) / 2 * k1 * a - 1j / 2 * k2 * a))
    h1[0, 1] = h1[1, 0].conj()  # 求共轭复数

    matrix = h0 + h1
    return matrix


def plot_bands_two_dimension(k1, k2, hamiltonian, filename='bansd_2D'):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    dim = hamiltonian(0, 0).shape[0]
    dim1 = k1.shape[0]
    dim2 = k2.shape[0]
    eigenvalue_k = np.zeros((dim2, dim1, dim))
    i0 = 0
    for k10 in k1:
        j0 = 0
        for k20 in k2:
            matrix0 = hamiltonian(k10, k20)
            eigenvalue, eigenvector = np.linalg.eig(matrix0)
            eigenvalue_k[j0, i0, :] = np.sort(np.real(eigenvalue[:]))
            j0 += 1
        i0 += 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    k1, k2 = np.meshgrid(k1, k2)
    for dim0 in range(dim):
        ax.plot_surface(k1, k2, eigenvalue_k[:, :, dim0], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


if __name__ == '__main__':
    hamiltonian0 = functools.partial(hamiltonian, M=0, t1=1, a=1 / sqrt(3))
    k1, k2 = np.linspace(-2 * pi, 2 * pi, 800), np.linspace(-2 * pi, 2 * pi, 800)
    plot_bands_two_dimension(k1, k2, hamiltonian0)
    print('finish')