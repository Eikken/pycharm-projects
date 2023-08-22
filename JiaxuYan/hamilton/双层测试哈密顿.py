# !/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   双层测试哈密顿.py    
@Time    :   2022/1/5 10:33  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import matplotlib.pyplot as plt
import cmath
import functools as ft


def hamiltonian(k, N):
    t = 1
    a = 1
    # 初始矩阵
    h = np.zeros((4 * N, 4 * N), dtype=complex)
    t0 = 0.2  # 层间跃迁
    V = 0.2  # 层间势能差
    for i in range(N):
        h[i * 4 + 0, i * 4 + 0] = V  # h00
        h[i * 4 + 1, i * 4 + 1] = V  # h11
        h[i * 4 + 2, i * 4 + 2] = -V  # h22
        h[i * 4 + 3, i * 4 + 3] = -V  # h33

        h[i * 4 + 0, i * 4 + 1] = -t * (1 + np.exp(1j * k * a))  # h01
        h[i * 4 + 1, i * 4 + 0] = -t * (1 + np.exp(-1j * k * a))  # h12
        h[i * 4 + 2, i * 4 + 3] = -t * (1 + np.exp(1j * k * a))  # h23
        h[i * 4 + 3, i * 4 + 2] = -t * (1 + np.exp(-1j * k * a))  # h32

        h[i * 4 + 0, i * 4 + 3] = -t0  # h03
        h[i * 4 + 3, i * 4 + 0] = -t0  # h30

    for i in range(N - 1):
        # 最近邻
        h[i * 4 + 1, (i + 1) * 4 + 0] = -t  # h14
        h[(i + 1) * 4 + 0, i * 4 + 1] = -t  # h05
        h[i * 4 + 3, (i + 1) * 4 + 2] = -t  # h36
        h[(i + 1) * 4 + 2, i * 4 + 3] = -t  # h27

    return h


def plot_bands_one_dimension(ky, hamilton0):
    dim = hamilton0(0).shape[0]
    dim1 = ky.shape[0]
    eigen_k = np.zeros((dim1, dim))
    i0 = 0
    for y0 in ky:
        matrix0 = hamilton0(y0)
        eigen_value, eigen_vector = np.linalg.eig(matrix0)
        eigen_k[i0, :] = np.sort(np.real(eigen_value[:]))
        i0 += 1
        print(y0)
    for d0 in range(dim):
        plt.plot(ky, eigen_k[:, d0], '-k', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    ham0 = ft.partial(hamiltonian, N=400)
    y = np.linspace(-np.pi, np.pi, 200)
    plot_bands_one_dimension(y, ham0)
    print('finish')