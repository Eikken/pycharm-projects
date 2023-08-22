#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   哈密顿量.py    
@Time    :   2021/5/8 16:08  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import matplotlib.pyplot as plt
from math import *
import cmath
import functools


def hamiltonian(k, N):
    h = np.zeros((4 * N, 4 * N)) * (1 + 0j)
    t = 1
    a = 1
    t0 = 0.2
    V = 0.2
    for i in range(N):
        h[i * 4 + 0, i * 4 + 0] = V
        h[i * 4 + 1, i * 4 + 1] = V
        h[i * 4 + 2, i * 4 + 2] = -V
        h[i * 4 + 3, i * 4 + 3] = -V

        h[i * 4 + 0, i * 4 + 1] = -t * (1 + cmath.exp(1j * k * a))
        h[i * 4 + 1, i * 4 + 0] = -t * (1 + cmath.exp(-1j * k * a))
        h[i * 4 + 2, i * 4 + 3] = -t * (1 + cmath.exp(1j * k * a))
        h[i * 4 + 3, i * 4 + 2] = -t * (1 + cmath.exp(-1j * k * a))

        h[i * 4 + 0, i * 4 + 3] = -t0
        h[i * 4 + 3, i * 4 + 0] = -t0

    for i in range(N - 1):
        # 最近邻
        h[i * 4 + 1, (i + 1) * 4 + 0] = -t
        h[(i + 1) * 4 + 0, i * 4 + 1] = -t
        h[i * 4 + 3, (i + 1) * 4 + 2] = -t
        h[(i + 1) * 4 + 2, i * 4 + 3] = -t

    return h


def plot_bands_one_dimension(k, hamiltonian, filename='bands_ID'):
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
    for dim0 in range(dim):
        plt.plot(k, eigenvalue_k[:, dim0], '-k', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    hamiltonian0 = functools.partial(hamiltonian, N=100)
    k = np.linspace(-pi, pi, 400)
    plot_bands_one_dimension(k, hamiltonian0)
