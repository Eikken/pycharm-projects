#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   知乎Haldane模型.py    
@Time    :   2022/6/15 14:38  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   H = h00 + (h01*e^ikr1 + h02*e^ikr2 + h03*e^ikr3 + h.c.)
'''


import numpy as np
import functools as ft
import matplotlib.pyplot as plt
import time


def HaldaneFunc(kx_, ky_,  a=1 / np.sqrt(3)):
    M = 2 / 3
    t1_ = 1
    t2_ = 1 / 3
    phi = np.pi / 4
    H = np.zeros((2, 2), dtype=complex)

    v1 = np.conj([[3 ** 0.5, 0]]).T  # 注意转置矩阵的坑[a, b].T != [[a, b]].T
    v2 = np.conj([[-3 ** 0.5 / 2, 3 / 2]]).T
    v3 = np.conj([[-3 ** 0.5 / 2, -3 / 2]]).T

    s0 = np.eye(2, dtype=int)
    sx = np.asmatrix([[0, 1], [1, 0]])

    sy = np.asmatrix([[0, -1j], [1j, 0]])

    sz = np.diag([1, -1])

    k_ = np.mat([kx_, ky_])
    H[0, 0] = t2_ * np.exp(1j * phi) * (np.exp(1j * k_ * v1) + np.exp(1j * k_ * v2) + np.exp(1j * k_ * v3))
    H[1, 1] = t2_ * np.exp(-1j * phi) * (np.exp(1j * k_ * v1) + np.exp(1j * k_ * v2) + np.exp(1j * k_ * v3))
    H[0, 1] = t1_ * (1 + np.exp(-1j * k_ * v2) + np.exp(1j * k_ * v3))
    H = M * sz + H + np.conj(H).T
    return H


def HaldanePlot(x_, y_, HaldaneHamilton):
    dim = HaldaneHamilton(0, 0).shape[0]

    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    dim2 = y_.shape[0]
    eigen_k = np.zeros((dim2, dim1, dim))
    i0 = 0
    for x0 in x_:
        j0 = 0
        for y0 in y_:
            matrix0 = HaldaneHamilton(x0, y0)
            eigenValue, eigenVector = np.linalg.eig(matrix0)
            # print(eigenValue)
            eigen_k[i0, j0, :] = np.sort(np.real(eigenValue[:]))
            j0 += 1
        i0 += 1

    print(eigen_k[0][0])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_, y_ = np.meshgrid(x_, y_)
    ax.plot_surface(x_, y_, eigen_k[:, :, 1], cmap='rainbow_r', linewidth=0,
                    antialiased=False)
    ax.plot_surface(x_, y_, eigen_k[:, :, 0], cmap='rainbow', linewidth=0,
                    antialiased=False)
    # ax.contour(x_, y_, eigen_k[:, :, 1], zdir='z', offset=0, cmap=plt.get_cmap('rainbow'))
    plt.title('Haldane')
    plt.show()

if __name__ == '__main__':
    t1 = time.time()
    num = 100
    # KX = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    # KY = np.linspace(-2 * np.pi, 2 * np.pi, 200)
    kx = np.linspace(-np.pi, np.pi, num)
    ky = np.linspace(-np.pi, np.pi, num)
    ham0 = ft.partial(HaldaneFunc)
    # print(ham0(0, 0).shape)

    HaldanePlot(kx, ky, ham0)
    t2 = time.time()

    print('finish', end=' ')
    print('use time %d s' % (t2 - t1))
