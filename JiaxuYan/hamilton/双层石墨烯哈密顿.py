#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   双层石墨烯哈密顿.py    
@Time    :   2021/12/29 10:35  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath
import functools


def hamiltonian(k, N):
    # 初始化为零矩阵
    h = np.zeros((4 * N, 4 * N), dtype=complex)
    h11 = np.zeros((4 * N, 4 * N), dtype=complex)  # 元胞内
    h12 = np.zeros((4 * N, 4 * N), dtype=complex)  # 元胞间

    t = 1
    a = 1
    t0 = 0.2  # 层间跃迁
    V = 0.1  # 层间的势能差为2V

    for i in range(N):
        h11[i * 2 + 0, i * 2 + 0] = V  # 对角线方向 : h00, h11 = V
        h11[i * 2 + 1, i * 2 + 1] = V

        h11[N * 2 + i * 2 + 0, N * 2 + i * 2 + 0] = -V
        h11[N * 2 + i * 2 + 1, N * 2 + i * 2 + 1] = -V

        h11[i * 2 + 0, i * 2 + 1] = -t
        h11[i * 2 + 1, i * 2 + 0] = -t

        h11[N * 2 + i * 2 + 0, N * 2 + i * 2 + 1] = -t
        h11[N * 2 + i * 2 + 1, N * 2 + i * 2 + 0] = -t

        h11[i * 2 + 0, N * 2 + i * 2 + 1] = -t0
        h11[N * 2 + i * 2 + 1, i * 2 + 0] = -t0

    for i in range(N - 1):
        h11[i * 2 + 1, (i + 1) * 2 + 0] = -t
        h11[(i + 1) * 2 + 0, i * 2 + 1] = -t

        h11[N * 2 + i * 2 + 1, N * 2 + (i + 1) * 2 + 0] = -t
        h11[N * 2 + (i + 1) * 2 + 0, N * 2 + i * 2 + 1] = -t

    for i in range(N):
        h12[i * 2 + 0, i * 2 + 1] = -t
        h12[N * 2 + i * 2 + 0, N * 2 + i * 2 + 1] = -t

    h = h11 + h12 * cmath.exp(-1j * k * a) + h12.transpose().conj() * cmath.exp(1j * k * a)  # 转置后共轭
    return h


def main():
    hamiltonian0 = functools.partial(hamiltonian, N=25)
    k = np.linspace(0, 2*pi, 100)
    plot_bands_one_dimension(k, hamiltonian0)


def plot_bands_one_dimension(k, hamiltonian):
    dim = hamiltonian(0).shape[0]
    dim_k = k.shape[0]
    eigenvalue_k = np.zeros((dim_k, dim))
    i0 = 0
    for k0 in k:
        matrix0 = hamiltonian(k0)
        eigenvalue, eigenvector = np.linalg.eig(matrix0)
        eigenvalue_k[i0, :] = np.sort(np.real(eigenvalue[:]))
        i0 += 1
        # print(k0)
    plt.figure(figsize=(8, 6), dpi=200)
    plt.title('TBG')
    for dim0 in range(dim):
        plt.plot(k, eigenvalue_k[:, dim0], '-k', linewidth=0.5)

    plt.xticks([])
    plt.show()


if __name__ == '__main__':
    # hamiltonian0 = functools.partial(hamiltonian, N=100)
    main()
    print('finish')