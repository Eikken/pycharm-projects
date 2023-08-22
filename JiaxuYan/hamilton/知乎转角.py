#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   知乎转角.py    
@Time    :   2022/7/2 7:48
@E-mail  :   iamwxyoung@qq.com
@Tips    :   仿照知乎网站 https://zhuanlan.zhihu.com/p/442528883
                matlab函数 kron
                格式 C=kron(A,B)   %A为m×n矩阵，B为p×q矩阵，则C为mp×nq矩阵。
'''
import time

import numpy as np
import functools as ft
import matplotlib.pyplot as plt


def HSLG_func(kx_, ky_, **kwargs):
    vf_ = kwargs['vf']
    k1_ = kwargs['kk']
    j1_ = kwargs['jj']
    cc_distance_ = kwargs['cc_distance']
    theta_ = kwargs['theta']
    b1m_ = 8 * np.pi * np.sin(np.abs(theta_)) / (cc_distance_ * 3) * np.array([1 / 2, -3 ** 0.5 / 2])
    b2m_ = 8 * np.pi * np.sin(np.abs(theta_)) / (cc_distance_ * 3) * np.array([1 / 2, 3 ** 0.5 / 2])
    kqx_ = kx_ - k1_ * b1m_[0] - j1_ * b2m_[0]
    kqy_ = ky_ - k1_ * b1m_[1] - j1_ * b2m_[1]

    h_ = -vf_ * (kqx_ ** 2 + kqy_ ** 2) ** 0.5 * np.array([[0, np.exp(1j * np.angle(kqx_ + 1j * kqy_) - theta_)],
                                                           [np.exp(-1j * np.angle(kqx_ + 1j * kqy_) - theta_), 0]])
    return h_


def HoffFunc(trc_):
    tr_ = trc_
    Hoff1 = np.zeros((2 * N, 2 * N))
    Hoff2 = np.zeros((2 * N, 2 * N))
    counter = 1
    for k2 in range(-tr_, tr_ + 1):
        for j2 in range(-tr_, tr_ + 1):
            for k3 in range(-tr_, tr_ + 1):
                for j3 in range(-tr_, tr_ + 1):
                    if k2 == k3 and j2 == j3:
                        # print(counter, '= =', k2, k3, j2, j3, (k3 + tr_) * s + j3 + tr_, (k2 + tr_) * s + j2 + tr_)
                        counter += 1
                        off1 = np.zeros((N, N))
                        off1[(k3 + tr_) * s + j3 + tr_][(k2 + tr_) * s + j2 + tr_] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Tb)
                        off2 = np.zeros((N, N))
                        off2[(k2 + tr_) * s + j2 + tr_][(k3 + tr_) * s + j3 + tr_] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Tb).T)
                    elif k2 == k3 and j3 + 1 == j2:
                        # print(counter, '= +', k2, k3, j2, j3, (k3 + tr_) * s + j3 + tr_, (k2 + tr_) * s + j2 + tr_)
                        counter += 1
                        off1 = np.zeros((N, N))
                        off1[(k3 + tr_) * s + j3 + tr_][(k2 + tr_) * s + j2 + tr_] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Ttr)
                        off2 = np.zeros((N, N))
                        off2[(k2 + tr_) * s + j2 + tr_][(k3 + tr_) * s + j3 + tr_] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttr).T)
                    elif k2 == k3-1 and j3 == j2:
                        # print(counter, '- =', k2, k3, j2, j3, (k3 + tr_) * s + j3 + tr_, (k2 + tr_) * s + j2 + tr_)
                        counter += 1
                        off1 = np.zeros((N, N))
                        off1[(k3 + tr_) * s + j3 + tr_][(k2 + tr_) * s + j2 + tr_] = 1
                        Hoff1 = Hoff1 + np.kron(off1, Ttl)
                        off2 = np.zeros((N, N))
                        off2[(k2 + tr_) * s + j2 + tr_][(k3 + tr_) * s + j3 + tr_] = 1
                        Hoff2 = Hoff2 + np.kron(off2, np.conj(Ttl).T)

    vHoff = np.kron(np.eye(2, k=1), Hoff1) + np.kron(np.eye(2, k=-1), Hoff2)  # 克罗内克积不符合交换律
    return vHoff


if __name__ == '__main__':
    time1 = time.time()
    ccDis = 1.42  # Å, nearest c-c bond length
    vf = 5.944  # eV·Å Fermi velocity
    phi = 2 * np.pi / 3
    agl = 0.521
    ylimt = 0.08  # 0.5°
    # ylimt = 1.75  # 5°
    # ylimt = 0.25  # 1.08°
    theta = np.deg2rad(agl)
    w1 = 0.110
    # w1 = 0

    Tb = w1 * np.ones((2, 2))
    Ttr = w1 * np.array([
        [np.exp(-1j * phi), 1],
        [np.exp(1j * phi), np.exp(-1j * phi)]
    ])
    Ttl = w1 * np.array([
        [np.exp(1j * phi), 1],
        [np.exp(-1j * phi), np.exp(1j * phi)]
    ])

    qb = 8 * np.pi * np.sin(theta / 2) / (ccDis * 3 * 3 ** 0.5) * np.array([0, -1])
    qtr = 8 * np.pi * np.sin(theta / 2) / (ccDis * 3 * 3 ** 0.5) * np.array([3 ** 0.5 / 2, 1 / 2])
    qtl = 8 * np.pi * np.sin(theta / 2) / (ccDis * 3 * 3 ** 0.5) * np.array([- 3 ** 0.5 / 2, 1 / 2])
    norm_qb = np.linalg.norm(qb)

    trc = 3  # truncation 截断值缩写命名
    s = 2 * trc + 1  # [-3, 3] 7个点
    N = s ** 2  # 49
    H = np.zeros((2 * 2 * N, 2 * 2 * N), dtype=complex)  # hamiltonian

    N1y = np.linspace(-1, 0, num=58) * norm_qb
    N1x = 0 * N1y

    N2y = np.linspace(0, 1, num=58) * norm_qb
    N2x = 0 * N2y

    N3y = np.linspace(1, -1 / 2, num=100) * norm_qb
    N3x = (-N3y + norm_qb) / 3 ** 0.5

    N4y = np.linspace(-1 / 2, -1, num=58) * norm_qb
    N4x = (N4y + norm_qb) * 3 ** 0.5

    Nx = [N1x, N2x, N3x, N4x]
    Ny = [N1y, N2y, N3y, N4y]

    eigen_k = np.zeros((274, 196))  # 274 = 路径上点的数量总和 ; 196 = 2*2*49
    vHf = HoffFunc(trc_=trc)
    eigIdx = 0

    #  ############ 以上的区间与变量都没有问题，所有的值已经确定，存在问题的是H_diag和H_off ###############

    for idx in range(4):
        kx = Nx[idx]
        ky = Ny[idx]
        for pointIndex in range(len(kx)):
            point_x, point_y = kx[pointIndex], ky[pointIndex]  # 点就是点，在A-B路径上的num个点,(x,y)是一一对应，点点对应的
            # 不能拆开也不能组合遍历， 参照 VASP kit计算
            # 每个点计算出一组特征值，特征值是几行就是几个点，196行就是point1的196个点投在DOS图上不同的高度
            cc = 0
            H_diag = np.zeros((2 * 2 * N, 2 * 2 * N), dtype=complex)  # 对角元 diagonal element
            for k0 in range(-trc, trc + 1):  # 走遍 tr = 3 的 [-3, 3] 组合 7*7 = 49
                for j0 in range(-trc, trc + 1):  # 基矢有2*49个,2代表上下两层（红蓝k点），最后还得乘2是因为矩阵元每个都是2×2的矩阵
                    tmp = np.zeros((2 * N, 2 * N))
                    tmp[cc][cc] = 1
                    hF = ft.partial(HSLG_func, vf=vf, kk=k0, jj=j0, cc_distance=ccDis, theta=theta / 2)
                    H_diag = H_diag + np.kron(tmp, hF(point_x, point_y))

                    cc += 1

            for k1 in range(-trc, trc + 1):
                for j1 in range(-trc, trc + 1):
                    tmp = np.zeros((2 * N, 2 * N))
                    tmp[cc][cc] = 1
                    hF = ft.partial(HSLG_func, vf=vf, kk=k1, jj=j1, cc_distance=ccDis, theta=-theta / 2)
                    H_diag = H_diag + np.kron(tmp, hF(point_x - qb[0], point_y - qb[1]))  # kx , ky

                    cc += 1

            H = H_diag + vHf
            eigenValue, eigenVector = np.linalg.eig(H)
            eigen_k[eigIdx, :] = np.sort(np.real(eigenValue[:]))
            eigIdx += 1
        # eigen_k 存储的是274*196 即每个坐标点垂直方向上算出196个特征值，196由3*3的截断面给出
        # 走过路径A-B-C-D-A

    plt.figure(figsize=(7, 5))
    routeList = np.linspace(1, 274, 274)
    for i in range(196):
        plt.plot(routeList, eigen_k[:, i], linewidth=1)


    plt.ylim(-ylimt, ylimt)
    lw = 0.3
    plt.axvline(58, color='gray', linestyle='--', linewidth=lw, label='K')
    plt.axvline(58*2, color='gray', linestyle='--', linewidth=lw, label='L')
    plt.axvline(100+58*2, color='gray', linestyle='--', linewidth=lw, label='M')
    plt.axhline(0, color='gray', linestyle='--', linewidth=lw, label='0')
    xk = [0, 58, 58*2, 100+58*2, 100+58*3]
    plt.xticks(xk, ["A", "B", "C", "D", "A"])
    # plt.legend()
    plt.title('band_dispersion_%.2f°' % agl)
    plt.show()

    time2 = time.time()
    print('finish use time %d s' % (time2 - time1))
