#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   我的动力学声子谱复述代码.py    
@Time    :   2023/2/13 9:20  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    np.size(a) 返回矩阵a中元素的个数
    np.size(a, axis=0) 返回a矩阵中的行数
    np.size(a, axis=1) 返回a矩阵中的列数
    np.matmul() 矩阵乘法，与.dot有所不同，可以广播

'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import distance


def R_phi(theta_):
    return np.array([[np.cos(theta_), np.sin(theta_), 0],
                     [-np.sin(theta_), np.cos(theta_), 0],
                     [0, 0, 1]])


def delta_theta(dis_):
    # dis_ = distance between the interacting atoms from a given atomic configuration corresponding to angle theta
    return 4 * epsilon * (156 * sigma ** 12 / dis_ ** 14 - 42 * sigma ** 6 / dis_ ** 8)


def dis_atom_a(a_, b_):
    return distance.cdist(a_, b_)


def fcm(*args, **kwargs):  # force constant matrix
    # [theta_, r_] = [i for i in args]
    # sp_ = kwargs['sp']  # sp_: sphere
    v_ = kwargs['v']  # v_: vector
    kA = []
    kB = []

    Phi_sp_1 = force_constant[0:3, 0:3]
    Phi_sp_2 = R_phi(np.pi / 2) @ force_constant[3:6, 3:6] @ np.linalg.inv(R_phi(np.pi / 2))
    # Phi_sp_2 = force_constant[3:6, 3:6]
    Phi_sp_3 = R_phi(np.pi) @ force_constant[6:9, 6:9] @ R_phi(np.pi).T
    # Phi_sp_4 = force_constant[9:12, 9:12]
    Phi_sp_4 = R_phi(19 / 180 * np.pi) @ force_constant[9:12, 9:12] @ R_phi(19 / 180 * np.pi).T
    # The force constant matrices for all atoms from four neighbor atomic spheres
    # can be found from Phi^(n) using the rotations around the Z axis by an angle ϕ in a clockwise direction.
    sigma_y = np.diag([1, -1, 1])  # the reflections in the XZ plane
    rb_ = []
    rt3 = np.sqrt(3)
    a_ = a

    U = np.array([[-np.cos(np.pi), -np.sin(np.pi), 0],
                  [np.sin(np.pi), np.cos(np.pi), 0],
                  [0, 0, 1]])
    # np.linalg.inv(U) @ r_ @ U
    # 通过旋转矩阵U可以将以A类原子为中心的近邻原子的相应的K矩阵转化成以B类原子为中心的
    # 第一遍处理的时候，把A转化为B取倒数
    # sp_ = 1 three atoms, B site
    for i in range(3):  # 2*np.pi/3
        r_ = R_phi(2 * np.pi / 3 * i) @ Phi_sp_1 @ R_phi(2 * np.pi / 3 * i).T
        kA.append(r_)
        kB.append(U @ r_ @ U.T)

    # sp_ = 2 six atoms, A site
    r_ = R_phi(0) @ Phi_sp_2 @ np.linalg.inv(R_phi(0))
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = sigma_y @ (R_phi(2 * np.pi / 3) @ Phi_sp_2 @ R_phi(2 * np.pi / 3).T) @ sigma_y.T
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = R_phi(2 * np.pi / 3) @ Phi_sp_2 @ R_phi(2 * np.pi / 3).T
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = sigma_y @ Phi_sp_2 @ sigma_y.T
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = R_phi(4 * np.pi / 3) @ Phi_sp_2 @ R_phi(4 * np.pi / 3).T
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = sigma_y @ (R_phi(4 * np.pi / 3) @ Phi_sp_2 @ R_phi(4 * np.pi / 3).T) @ sigma_y.T
    kA.append(r_)
    kB.append(U @ r_ @ U.T)

    # sp_ = 3 three atoms, B site
    for i in range(3):  # 2*np.pi/3
        r_ = R_phi(2 * np.pi / 3 * i) @ Phi_sp_3 @ R_phi(2 * np.pi / 3 * i).T
        kA.append(r_)
        kB.append(U @ r_ @ U.T)

    # sp_ = 4 six atoms, B site, two angle types
    r_ = sigma_y @ (R_phi(0) @ Phi_sp_4 @ np.linalg.inv(R_phi(0))) @ np.linalg.inv(sigma_y)
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = R_phi(0) @ Phi_sp_4 @ np.linalg.inv(R_phi(0))
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = sigma_y @ (R_phi(4 * np.pi / 3) @ Phi_sp_4 @ np.linalg.inv(R_phi(4 * np.pi / 3))) @ np.linalg.inv(sigma_y)
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = R_phi(2 * np.pi / 3) @ Phi_sp_4 @ np.linalg.inv(R_phi(2 * np.pi / 3))
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = sigma_y @ (R_phi(2 * np.pi / 3) @ Phi_sp_4 @ np.linalg.inv(R_phi(2 * np.pi / 3))) @ np.linalg.inv(sigma_y)
    kA.append(r_)
    kB.append(U @ r_ @ U.T)
    r_ = R_phi(4 * np.pi / 3) @ Phi_sp_4 @ np.linalg.inv(R_phi(4 * np.pi / 3))
    kA.append(r_)
    kB.append(U @ r_ @ U.T)

    # sp_ = 1 three atoms, B site
    # hij vector j=[i for i in range(1, 7)], vij = vector i2j,这个文献他Fig2按顺时针转,这样角度就毫无顺序了就
    # return value 分为: 0,3; 3,9; 9,12; 12,18
    v11 = np.array([1, 0, 0]) * a_
    v12 = np.array([-1 / 2, -rt3 / 2, 0]) * a_
    v13 = np.array([-1 / 2, rt3 / 2, 0]) * a_
    FirA = [v11, v12, v13]
    FirB = list(-1 * np.array(FirA))
    # sp_ = 2 six atoms, A site
    v21 = np.array([0, rt3, 0]) * a_
    v22 = np.array([3 / 2, rt3 / 2, 0]) * a_
    v23 = np.array([3 / 2, -rt3 / 2, 0]) * a_
    v24 = np.array([0, -rt3, 0]) * a_
    v25 = np.array([-3 / 2, -rt3 / 2, 0]) * a_
    v26 = np.array([-3 / 2, rt3 / 2, 0]) * a_
    SecA = [v21, v22, v23, v24, v25, v26]
    SecB = list(-1 * np.array(SecA))
    # sp_ = 3 three atoms, B site
    v31 = np.array([-2, 0, 0]) * a_
    v32 = np.array([1, rt3, 0]) * a_
    v33 = np.array([1, -rt3, 0]) * a_
    ThiA = [v31, v32, v33]
    ThiB = list(-1 * np.array(ThiA))
    # sp_ = 4 six atoms, B site, two angle types
    v41 = np.array([5 / 2, rt3 / 2, 0]) * a_
    v42 = np.array([5 / 2, -rt3 / 2, 0]) * a_
    v43 = np.array([-1 / 2, -3 * rt3 / 2, 0]) * a_
    v44 = np.array([-2, -rt3, 0]) * a_
    v45 = np.array([-2, rt3, 0]) * a_
    v46 = np.array([-1 / 2, 3 * rt3 / 2, 0]) * a_
    FouA = [v41, v42, v43, v44, v45, v46]
    FouB = list(-1 * np.array(FouA))

    KAB1, KBA1 = kA[:3], kB[:3]
    KAA, KBB = kA[3:9], kB[3:9]
    KAB3, KBA3 = kA[9:12], kB[9:12]
    KAB4, KBA4 = kA[12:18], kB[12:18]

    DAAs, DBBs, DBAs, DABs = 0, 0, 0, 0
    for i in range(3):
        DAAs = DAAs + KAA[i] * np.exp(1j * np.matmul(v_, SecA[i])) + \
               KAA[i + 3] * np.exp(1j * np.matmul(v_, SecA[i + 3]))
        DBBs = DAAs
        DABs = DABs + KAB1[i] * np.exp(1j * np.matmul(v_, FirA[i])) + \
               KAB3[i] * np.exp(1j * np.matmul(v_, ThiA[i])) + \
               KAB4[i] * np.exp(1j * np.matmul(v_, FouA[i], )) + \
               KAB4[i + 3] * np.exp(1j * np.matmul(v_, FouA[i + 3]))
        DBAs = DABs.T.conj()
    D = np.zeros([6, 6], dtype=np.complex)

    D[0:3, 3:6] = -DABs
    D[3:6, 0:3] = -DBAs
    D[0:3, 0:3] = sum([sum(KAB1), sum(KAA), sum(KAB3), sum(KAB4)]) - DAAs
    D[3:6, 3:6] = sum([sum(KBA1), sum(KAA), sum(KBA3), sum(KBA4)]) - DBBs
    # return kA, kB
    # print(D)
    return D


if __name__ == '__main__':
    # start here
    time1 = time.time()
    a = 1.42e-10
    m = 1.99e-26
    # a = 0.142  # nm
    # m = 1.99
    epsilon = 4.6  # meV
    sigma = 0.3276  # nm

    n = 100
    # force_constant = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])
    force_constant = np.diag([398.7, 172.8, 98.9,
                              72.9, -46.1, -8.2,
                              -26.4, 33.1, 5.8,
                              1.0, 7.9, -5.2])
    # 构建AB位点的rotate关系,前缀表示当前层,文献中是顺时针转的

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 6])  # 解的矩阵

    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值
            kx = i * 2 * np.pi / 3 / a / (n * int(10 * 3 ** 0.5))  #
            ky = 0
        elif i < (10 + int(10 * 3 ** 0.5)) * n:
            kx = 2 * np.pi / 3 / a
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a / 3 ** 0.5
        else:
            kx = 2 * np.pi / 3 / a - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a
            ky = kx / 3 ** 0.5
        k = np.array([kx, ky, 0])  # 得到k值，带入D矩阵
        dm = fcm(v=k)
        w, t = np.linalg.eig(dm)
        w = list(w)
        w.sort()
        result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存
    s = 3 ** 0.5
    xk = [0, s, s + 1, s + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(3 ** 0.5 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(4, 5))
    plt.plot(kk, result, c="r")
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    plt.xlim(0, 4.75)
    # plt.ylim(0, 1e14)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(s, color='gray', linestyle='--')
    plt.axvline(s + 1, color='gray', linestyle='--')

    plt.show()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
