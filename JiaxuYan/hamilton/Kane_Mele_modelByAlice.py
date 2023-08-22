#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Kane_Mele_modelByAlice.py    
@Time    :   2022/6/20 17:22  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   e^ikr 这里的k和r 画一维就是一维[x]，画二维就是二维[x, y]
             向量的积就是一维，所有的算完都是一个一维的数据。
             点乘，也叫数量积。结果是一个向量在另一个向量方向上投影的长度，是一个标量。
             点积公式： x1*y1+x2*y2+……xn*yn  （n维向量）
             叉乘，也叫向量积。结果是一个和已有两个向量都垂直的向量（法向量）。
             叉乘公式：
             |i  j  k |
             |a1 b1 c1|
             |a2 b2 c2|
             =(b1c2-b2c1, c1a2-a1c2, a1b2-a2b1)
'''


import numpy as np
import functools as ft
import matplotlib.pyplot as plt
import time


def KaneMeleFunc(kx_, ky_):
    t_ = -1
    gamma_v = 0.1 * t_  # 质量项，打开带隙，相当于M
    gamma_so = 0.06*t_  # 次近邻自旋轨道耦合，相当于t2
    gamma_r = 0.05*t_  # Rashba类型自旋轨道耦合项
    state = 2

    sx_values = np.array([1, 0])
    sxn_values = np.max(sx_values) + 1  # 网上学的 one-hot 矩阵形式
    s0 = np.eye(2, dtype=int)
    sx = np.eye(sxn_values)[sx_values]
    sy = np.asmatrix([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    v1 = np.conj([[3 ** 0.5, 0]]).T
    v2 = np.conj([[-3 ** 0.5 / 2, 3 / 2]]).T
    v3 = np.conj([[-3 ** 0.5 / 2, -3 / 2]]).T

    # 把state换成2一样可以说清楚
    k_ = np.mat([kx_, ky_])

    H0 = np.zeros((4, 4), dtype=complex)
    # H0 = H0 + np.kron(gamma_v*s0, sz)
    H0 = H0 + np.kron(sz, gamma_v * s0)
    H0[0:state, state:state * 2] = t_ * s0 + 1j * gamma_r * (-sx)
    H0[state:state * 2, 0:state] = t_ * s0 + 1j * gamma_r * sx
    # print(H0)  # 不确定此处的H0矩阵是否描述正确

    H1 = np.zeros((4, 4), dtype=complex)
    H1 = H1 + np.kron(sz, -1j*gamma_so*sz)
    H1 = H1*np.exp(1j*k_*v1)[0, 0]

    H2 = np.zeros((4, 4), dtype=complex)
    H2[0:state, 0:state] = -1j * gamma_so * sz
    H2[state:state * 2, state:state * 2] = 1j * gamma_so * sz
    H2[state:state * 2, 0:state] = t_ * s0 + 1j * gamma_r * (-0.5 * sx - 3 ** 0.5 / 2 * sy)
    H2 = H2*np.exp(1j*k_*v2)[0, 0]

    H3 = np.zeros((4, 4), dtype=complex)
    H3[0:state, 0:state] = -1j * gamma_so * sz
    H3[state:state * 2, state:state * 2] = 1j * gamma_so * sz
    H3[0:state, state:state * 2] = t_ * s0 + 1j * gamma_r * (0.5 * sx - 3 ** 0.5 / 2 * sy)
    H3 = H3*np.exp(1j*k_*v3)[0, 0]

    # H = H0 + H1 + H1.T + H2 + H2.T + H3 + H3.T  # 注意此处为厄米共轭
    H = H0 + H1 + np.conj(H1).T + H2 + np.conj(H2).T + H3 + np.conj(H3).T
    return H


def KaneMelePlot(x_, y_, KaneMeleHamiltonian):
    dim = KaneMeleHamiltonian(0, 0).shape[0]
    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    dim2 = y_.shape[0]
    eigen_k = np.zeros((dim2, dim1, dim))
    i0 = 0
    for x0 in x_:
        j0 = 0
        for y0 in y_:
            matrix0 = KaneMeleHamiltonian(x0, y0)
            eigenValue, eigenVector = np.linalg.eig(matrix0)
            eigen_k[i0, j0, :] = np.sort(np.real(eigenValue[:]))
            j0 += 1
        i0 += 1

    fig = plt.figure()
    print(eigen_k.shape)
    ax = fig.gca(projection='3d')
    x_, y_ = np.meshgrid(x_, y_)
    # ax.plot_surface(x_, y_, eigen_k[:, :, 0], cmap='rainbow', linewidth=0,
    #                 antialiased=False)
    ax.plot_surface(x_, y_, eigen_k[:, :, 1], cmap='rainbow', linewidth=0,
                    antialiased=False)

    ax.plot_surface(x_, y_, eigen_k[:, :, 2], cmap='rainbow_r', linewidth=0,
                    antialiased=False)
    # ax.plot_surface(x_, y_, eigen_k[:, :, 3], cmap='rainbow', linewidth=0,
    #                 antialiased=False)
    # ax.contour(x_, y_, eigen_k[:, :, 1], zdir='z', offset=0, cmap=plt.get_cmap('rainbow'))
    plt.title('Kane_Mele')
    plt.show()


if __name__ == '__main__':
    t1 = time.time()
    num = 200
    kx = np.linspace(-np.pi, np.pi, num)
    ky = np.linspace(-np.pi, np.pi, num)

    ham0 = ft.partial(KaneMeleFunc)
    KaneMelePlot(kx, ky, ham0)
    t2 = time.time()

    print('finish', end=' ')
    print('use time %d s' % (t2-t1))
