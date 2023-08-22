#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   知乎边态大晶胞模型.py    
@Time    :   2022/6/16 10:48  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   H = np.zeros((2, 2), dtype=complex)
            计算时注意矩阵的 dtype 虚实特性，否则一些数据可能被省去导致计算误差
            matlab中 M.' 是矩阵转置， M'是矩阵的厄米共轭转置
'''

import numpy as np
import functools as ft
import matplotlib.pyplot as plt


def HaldanePlusFunc(kx_):
    M = 2 / 3
    t1_ = 1
    t2_ = 1 / 3
    phi = np.pi / 4
    N = 20

    s0 = np.eye(2, dtype=int)
    sx = np.asmatrix([[0, 1], [1, 0]])
    sy = np.asmatrix([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    T0 = np.zeros((4, 4), dtype=complex)
    T0 = T0 + np.kron(s0, M * sz) + t1_ * np.eye(4, k=1) + t1_ * np.eye(4, k=-1)
    T0[0, 2] = t2_ * np.exp(-1j * phi)
    T0[2, 0] = t2_ * np.exp(1j * phi)
    T0[1, 3] = t2_ * np.exp(-1j * phi)
    T0[3, 1] = t2_ * np.exp(1j * phi)

    Tx = np.zeros((4, 4), dtype=complex)
    Tx = Tx + t2_ * np.exp(1j * phi) * np.diag([1, -1, 1, -1])  # Tx 方向
    Tx[1, 0] = t1_
    Tx[2, 3] = t1_
    Tx[2, 0] = t2_ * np.exp(-1j * phi)
    Tx[1, 3] = t2_ * np.exp(1j * phi)

    Ty = np.zeros((4, 4), dtype=complex)
    Ty[2, 0] = t2_ * np.exp(1j * phi)
    Ty[3, 1] = t2_ * np.exp(1j * phi)
    Ty[3, 0] = t1_

    Txy = np.zeros((4, 4), dtype=complex)
    Txy[1, 3] = t2_ * np.exp(1j * phi)

    Tyx = np.zeros((4, 4), dtype=complex)
    Tyx[2, 0] = t2_ * np.exp(-1j * phi)

    # np.array([1, 2, 0, 1, 2])  # 纵向量位置为1，其余位置为0

    H00 = np.kron(np.eye(N, k=1), Ty)  # 克罗内克乘积
    H00 = np.kron(np.eye(N), T0) + H00 + np.conj(H00).T
    H01 = np.kron(np.eye(N), Tx) + \
          np.kron(np.eye(N, k=1), Tyx) + \
          np.kron(np.eye(N, k=-1), Txy)
    H = H00 + H01*np.exp(1j*kx_*np.pi) + np.conj(H01).T*np.exp(-1j*kx_*np.pi)
    return H


def HaldanePlusPlot(x_, HaldanePlusHamilton):
    dim = HaldanePlusHamilton(0).shape[0]
    print(dim)
    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    eigen_k = np.zeros((dim1, dim))
    i0 = 0
    for x0 in x_:
        matrix0 = HaldanePlusHamilton(x0)
        eigenValue, eigenVector = np.linalg.eig(matrix0)
        eigen_k[i0, :] = np.sort(np.real(eigenValue[:]))
        i0 += 1
    plt.figure(figsize=(7, 4))
    plt.plot(x_, eigen_k, linewidth=0.5, color='black')
    plt.title('HaldanePlus')
    plt.show()


if __name__ == '__main__':
    num = 200
    # KX = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    # KY = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    kx = np.linspace(0, 2, num)
    ky = np.linspace(-np.pi, np.pi, num)
    ham1 = ft.partial(HaldanePlusFunc)
    HaldanePlusPlot(kx, ham1)
    print('finish')
