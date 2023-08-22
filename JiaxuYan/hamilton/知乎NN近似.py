#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   知乎NN近似.py    
@Time    :   2022/1/4 19:26  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import functools as ft
import matplotlib.pyplot as plt


def NN_hamiltonian(kx, ky, t, a=1 / np.sqrt(3)):
    h0 = np.zeros((2, 2), dtype=complex)
    h1 = np.zeros((2, 2), dtype=complex)
    h0[0, 0], h0[1, 1] = 0, 0
    g3 = np.sqrt(3)
    h1[0, 1] = t * (3 + 2 * np.cos(g3 * a * kx) + 4 * np.cos(g3 / 2 * kx * a) * np.cos(
        3 / 2 * ky * a)) ** 0.5
    # h1[0, 1] = t * (np.exp(1j*a*g3/2*kx-1j*a*3/2*ky)+np.exp(-1j*a*g3/2*kx-1j*a*3/2*ky)+1)
    h1[1, 0] = h1[0, 1].conj()
    matrix = h0 + h1
    return matrix


def plot_X_Y_Bands(x_, y_, NN_ham):
    dim = NN_ham(0, 0).shape[0]
    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    dim2 = y_.shape[0]
    eigen_k = np.zeros((dim2, dim1, dim))
    # print(dim, dim1, dim2)

    i0 = 0
    for x0 in x_:
        j0 = 0
        for y0 in y_:
            matrix0 = NN_ham(x0, y0)
            eigenValue, eigenVector = np.linalg.eig(matrix0)
            # print(eigenValue)
            eigen_k[i0, j0, :] = np.sort(np.real(eigenValue[:]))
            j0 += 1
        i0 += 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_, y_ = np.meshgrid(x_, y_)
    # for dim0 in range(dim):
    #     ax.plot_surface(x_, y_, eigen_k[:, :, dim0], cmap='rainbow', linewidth=0,
    #                     antialiased=False)
    ax.plot_surface(x_, y_, eigen_k[:, :, 1], cmap='rainbow_r', linewidth=0,
                    antialiased=False)
    # ax.contour(x_, y_, eigen_k[:, :, 1], zdir='z', offset=0, cmap=plt.get_cmap('rainbow'))
    ax.plot_surface(x_, y_, eigen_k[:, :, 0], cmap='rainbow', linewidth=0,
                    antialiased=False)
    plt.title('NN')
    plt.show()


if __name__ == '__main__':
    constant_a = 1.42
    ham0 = ft.partial(NN_hamiltonian, t=1)
    KX = np.linspace(-2*np.pi, 2*np.pi, 200)
    KY = np.linspace(-2*np.pi, 2*np.pi, 200)
    plot_X_Y_Bands(KX, KY, ham0)
    print('finish')