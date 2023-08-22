#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Kane_Mele_1D.py    
@Time    :   2022/6/21 10:30  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   完成Kane Mele model的1D图像plot
             https://zhuanlan.zhihu.com/p/359578693
             e^ikr 这里的k和r 画一维就是一维[x]，画二维就是二维[x, y]
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


def KaneMele1DFunc(kx_):
    t_ = -1
    gamma_v = 0.1 * t_  # 质量项，打开带隙，相当于M
    gamma_so = 0.06 * t_  # 次近邻自旋轨道耦合，相当于t2
    gamma_r = 0.05 * t_  # Rashba类型自旋轨道耦合项
    state = 2  #
    N = 20

    sx_values = np.array([1, 0])
    sxn_values = np.max(sx_values) + 1  # one-hot 矩阵形式
    s0 = np.eye(2, dtype=int)
    sx = np.eye(sxn_values)[sx_values]  # [ [0, 1], [1, 0] ]
    sy = np.asmatrix([[0, -1j], [1j, 0]])
    sz = np.diag([1, -1])

    T0 = np.zeros((state*4, state*4), dtype=complex)
    T0[0:state * 1, state * 1:state * 2] = 1j*gamma_r*(0.5*sx + 3**0.5/2*sy) + t_*s0
    T0[state * 1:state * 2, state * 2:state * 3] = 1j*gamma_r*sx + t_*s0
    T0[state * 2:state * 3, state * 3:state * 4] = 1j*gamma_r*(0.5*sx - 3**0.5/2*sy) + t_*s0
    T0[0:state * 1, state * 2:state * 3] = 1j*gamma_so*sz
    T0[state * 1:state * 2, state * 3:state * 4] = 1j*gamma_so*sz
    kron1 = np.kron(np.eye(2), sz)  # 两次克罗内克积
    kron2 = np.kron(kron1, gamma_v*np.eye(2))  #
    T0 = kron2 + T0 + np.conj(T0).T

    Tx = np.zeros((state*4, state*4), dtype=complex)
    Tx[0:state * 1, 0:state * 1] = -1j * gamma_so * sz
    Tx[state * 1:state * 2, state * 1:state * 2] = 1j * gamma_so * sz
    Tx[state * 2:state * 3, state * 2:state * 3] = -1j * gamma_so * sz
    Tx[state * 3:state * 4, state * 3:state * 4] = 1j * gamma_so * sz
    Tx[state * 1:state * 2, 0:state * 1] = 1j * gamma_r * (-0.5*sx + 3**0.5/2*sy) + t_*s0
    Tx[state * 2:state * 3, 0:state * 1] = 1j * gamma_so * sz
    Tx[state * 1:state * 2, state * 3:state * 4] = -1j * gamma_so * sz
    Tx[state * 2:state * 3, state * 3:state * 4] = 1j * gamma_r * (0.5*sx + 3**0.5/2*sy) + t_*s0

    Ty = np.zeros((state*4, state*4), dtype=complex)
    Ty[state * 2:state * 3, 0:state * 1] = -1j * gamma_so * sz
    Ty[state * 3:state * 4, 0:state * 1] = 1j * gamma_r * sx + t_*s0
    Ty[state * 3:state * 4, state * 1:state * 2] = -1j * gamma_so * sz

    Txy = np.zeros((state*4, state*4), dtype=complex)  # Txy & Tyx 不是conj关系
    Txy[state * 1:state * 2, state * 3:state * 4] = -1j * gamma_so * sz

    Tyx = np.zeros((state*4, state*4), dtype=complex)
    Tyx[state * 2:state * 3, 0:state * 1] = 1j * gamma_so * sz

    H00 = np.kron(np.eye(N, k=1), Ty)
    H00 = np.kron(np.eye(N), T0) + H00 + np.conj(H00).T
    H01 = np.kron(np.eye(N), Tx) + np.kron(np.eye(N, k=1), Tyx) + np.kron(np.eye(N, k=-1), Txy)

    H = H00 + H01*np.exp(1j*kx_*np.pi) + np.conj(H01).T * np.exp(-1j*kx_*np.pi)

    return H


def KaneMele1DPlot(x_, KaneMele1DHamiltonian):
    dim = KaneMele1DHamiltonian(0).shape[0]
    dim1 = x_.shape[0]  # x_.shape[0] 等价于 len(x_)
    eigen_k = np.zeros((dim1, dim))
    i0 = 0
    for x0 in x_:
        matrix0 = KaneMele1DHamiltonian(x0)
        eigenValue, eigenVector = np.linalg.eig(matrix0)
        eigen_k[i0, :] = np.sort(np.real(eigenValue[:]))
        i0 += 1
    # plt.figure(figsize=(7, 4))
    plt.plot(x_, eigen_k, linewidth=0.5, color='black')
    # plt.ylim((-1, 1))
    plt.title('KaneMele1D')
    plt.show()


if __name__ == '__main__':
    t1 = time.time()
    num = 200  # 分割点数
    kx = np.linspace(0, 2, num)

    ham0 = ft.partial(KaneMele1DFunc)
    KaneMele1DPlot(kx, ham0)
    t2 = time.time()

    print('finish', end=' ')
    print('use time %d s' % (t2-t1))
