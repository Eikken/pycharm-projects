#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   TBG_dynamic_matrix.py    
@Time    :   2023/2/16 17:42  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance


def R_phi(theta_):
    return np.array([[np.cos(theta_), np.sin(theta_), 0],
                     [-np.sin(theta_), np.cos(theta_), 0],
                     [0, 0, 1]])


def project_rtd(a_, b_):  # R(θ) is the distance of two atoms
    # 13年文献的方法
    # dis_ = distance between the interacting atoms from a given atomic configuration corresponding to angle theta
    # 一行代码，大有来头
    epsilon = 4.6  # meV
    sigma = 0.3276  # nm
    a_, b_ = a_[:, 0], b_[:, 0]
    rtheta = b_ - a_
    ralpha = rtheta[0]
    rbeta = rtheta[1]
    norm_r = np.linalg.norm(rtheta)
    dis_ = distance.cdist([a_], [b_]).min()
    return 4 * epsilon * (156 * sigma ** 12 / dis_ ** 14 - 42 * sigma ** 6 / dis_ ** 8) * ralpha * rbeta / norm_r ** 2


def FCMs(a_, b_):  # R(θ) is the distance of two atoms
    # 新的层间力常数构建方法
    A_ = 573.76
    B_ = 0.05
    a_, b_ = a_[:, 0], b_[:, 0]
    rtheta = b_ - a_  # rtheta = rj - ri
    ralpha = rtheta[0]
    rbeta = rtheta[1]
    norm_r = np.linalg.norm(rtheta)
    delta_r = A_ * np.exp(-rtheta / B_)
    return -delta_r * ralpha * rbeta / norm_r ** 2


def fcm(*args, **kwargs):  # force constant matrix
    # sp_: sphere
    theta_ = kwargs['theta']
    v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)  # v_: vector 0;  w_ : vector w_0.5
    # w_ = np.c_[kwargs['v'].reshape(1, 2), d*a].reshape(1, 3)
    w_ = v_
    kA = []
    kB = []
    kA_ = []  # kA is A' atom
    kB_ = []
    rt3 = 3 ** 0.5
    # 在这里，我们认为也要对13年文献的初始force constant转到相对初始的张量状态
    # 旋转矩阵的逆矩阵等于旋转矩阵的转置   inv.(U) == U.T
    Phi_sp_1 = force_constant[0:3, 0:3]  # 层1不需要转, 或者说转Phi(0)， 严格按照13年FIG2的顺序对应关系
    Phi_sp_2 = R_phi(np.pi / 2) @ force_constant[3:6, 3:6] @ R_phi(np.pi / 2).T  # 层2转 π/2
    Phi_sp_3 = R_phi(np.pi) @ force_constant[6:9, 6:9] @ R_phi(np.pi).T  # 层3转 π
    Phi_sp_4_up = R_phi(-np.arctan(rt3 / 5)) @ force_constant[9:12, 9:12] @ R_phi(-np.arctan(rt3 / 5)).T
    # 层4上面转 arctan(rt3/5)
    Phi_sp_4_down = R_phi(-2 * np.pi + np.arctan(rt3 / 5)) @ force_constant[9:12, 9:12] @ R_phi(
        -2 * np.pi + np.arctan(rt3 / 5)).T
    # 层4下面转 2*π-arctan(rt3/5)
    # The force constant matrices for all atoms from four neighbor atomic spheres can be found from Phi^(n) using the
    # rotations around the Z axis by an angle ϕ in a clockwise direction.
    sigma_y = np.diag([1, -1, 1])  # the reflections in the XZ  文献中给出的sigma矩阵
    a_ = a

    # 来自旋转角度theta的作用部分
    Phi_up_1 = R_phi(theta_) @ Phi_sp_1 @ R_phi(theta_).T
    Phi_up_2 = R_phi(theta_) @ Phi_sp_2 @ R_phi(theta_).T
    Phi_up_3 = R_phi(theta_) @ Phi_sp_3 @ R_phi(theta_).T
    Phi_up_4_u = R_phi(theta_) @ Phi_sp_4_up @ R_phi(theta_).T
    Phi_up_4_d = R_phi(theta_) @ Phi_sp_4_down @ R_phi(theta_).T
    # Phi_up_1 = Phi_sp_1
    # Phi_up_2 = Phi_sp_2
    # Phi_up_3 = Phi_sp_3
    # Phi_up_4_u = Phi_sp_4_up
    # Phi_up_4_d = Phi_sp_4_down

    U = R_phi(np.pi)
    # 通过旋转矩阵U可以将以A类原子为中心的近邻原子的相应的K矩阵转化成以B类原子为中心的
    # sp_ = 1 three atoms, B site
    for im in range(3):  # 转2π/3
        r_ = R_phi(2 * np.pi / 3 * im) @ Phi_sp_1 @ R_phi(2 * np.pi / 3 * im).T
        kA.append(r_)
        kB.append(U.T @ r_ @ U)
        # 增加旋转层的Phi ij向量
        ra_ = R_phi(2 * np.pi / 3 * im) @ Phi_up_1 @ R_phi(2 * np.pi / 3 * im).T
        kA_.append(ra_)
        kB_.append(U.T @ ra_ @ U)

    # sp_ = 2 six atoms, A site
    for im in range(6):
        r_ = R_phi(im * np.pi / 3) @ Phi_sp_2 @ R_phi(im * np.pi / 3).T
        kA.append(r_)
        kB.append(U.T @ r_ @ U)

        ra_ = R_phi(im * np.pi / 3) @ Phi_up_2 @ R_phi(im * np.pi / 3).T
        kA_.append(ra_)
        kB_.append(U.T @ ra_ @ U)
    # sp_ = 3 three atoms, B site
    for im in range(3):  # 2*np.pi/3
        r_ = R_phi(2 * np.pi / 3 * im) @ Phi_sp_3 @ R_phi(2 * np.pi / 3 * im).T
        kA.append(r_)
        kB.append(U.T @ r_ @ U)

        ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_3 @ R_phi(im * 2 * np.pi / 3).T
        kA_.append(ra_)
        kB_.append(U.T @ ra_ @ U)
    # sp_ = 4 six atoms, B site, two angle types
    for im in range(3):
        r_ = R_phi(im * 2 * np.pi / 3) @ Phi_sp_4_up @ R_phi(im * 2 * np.pi / 3).T
        kA.append(r_)
        kB.append(U.T @ r_ @ U)

        r_ = R_phi(im * 2 * np.pi / 3) @ Phi_sp_4_down @ R_phi(im * 2 * np.pi / 3).T
        kA.append(r_)
        kB.append(U.T @ r_ @ U)

        ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_4_u @ R_phi(im * 2 * np.pi / 3).T
        kA_.append(ra_)
        kB_.append(U.T @ ra_ @ U)

        ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_4_d @ R_phi(im * 2 * np.pi / 3).T
        kA_.append(ra_)
        kB_.append(U.T @ ra_ @ U)

    # return value 分为: 0,3; 3,9; 9,12; 12,18
    v11 = np.array([1, 0, 0]).reshape(3, 1) * a_
    v12 = np.array([-1 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
    v13 = np.array([-1 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
    FirA = [v11, v12, v13]
    FirB = list(-1 * np.array(FirA))
    # sp_ = 2 six atoms, A site
    v21 = np.array([0, rt3, 0]).reshape(3, 1) * a_
    v22 = np.array([3 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
    v23 = np.array([3 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
    v24 = np.array([0, -rt3, 0]).reshape(3, 1) * a_
    v25 = np.array([-3 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
    v26 = np.array([-3 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
    SecA = [v21, v22, v23, v24, v25, v26]
    SecB = list(-1 * np.array(SecA))
    # sp_ = 3 three atoms, B site
    v31 = np.array([-2, 0, 0]).reshape(3, 1) * a_
    v32 = np.array([1, rt3, 0]).reshape(3, 1) * a_
    v33 = np.array([1, -rt3, 0]).reshape(3, 1) * a_
    ThiA = [v31, v32, v33]
    ThiB = list(-1 * np.array(ThiA))
    # sp_ = 4 six atoms, B site, two angle types
    v41 = np.array([5 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
    v42 = np.array([5 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
    v43 = np.array([-1 / 2, -3 * rt3 / 2, 0]).reshape(3, 1) * a_
    v44 = np.array([-2, -rt3, 0]).reshape(3, 1) * a_
    v45 = np.array([-2, rt3, 0]).reshape(3, 1) * a_
    v46 = np.array([-1 / 2, 3 * rt3 / 2, 0]).reshape(3, 1) * a_
    FouA = [v41, v42, v43, v44, v45, v46]
    FouB = list(-1 * np.array(FouA))

    # layer 2 vectors
    w11 = np.dot(R_phi(theta_), np.array([1, 0, d]).reshape(3, 1)) * a_
    w12 = np.dot(R_phi(theta_), np.array([-1 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
    w13 = np.dot(R_phi(theta_), np.array([-1 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
    FirA_up = [w11, w12, w13]
    FirB_up = list(-1 * np.array(FirA_up))
    # sp_ = 2 six atoms, A site
    w21 = np.dot(R_phi(theta_), np.array([0, rt3, d]).reshape(3, 1)) * a_
    w22 = np.dot(R_phi(theta_), np.array([3 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
    w23 = np.dot(R_phi(theta_), np.array([3 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
    w24 = np.dot(R_phi(theta_), np.array([0, -rt3, d]).reshape(3, 1)) * a_
    w25 = np.dot(R_phi(theta_), np.array([-3 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
    w26 = np.dot(R_phi(theta_), np.array([-3 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
    SecA_up = [w21, w22, w23, w24, w25, w26]
    SecB_up = list(-1 * np.array(SecA_up))
    # sp_ = 3 three atoms, B site
    w31 = np.dot(R_phi(theta_), np.array([-2, 0, d]).reshape(3, 1)) * a_
    w32 = np.dot(R_phi(theta_), np.array([1, rt3, d]).reshape(3, 1)) * a_
    w33 = np.dot(R_phi(theta_), np.array([1, -rt3, d]).reshape(3, 1)) * a_
    ThiA_up = [w31, w32, w33]
    ThiB_up = list(-1 * np.array(ThiA_up))
    # sp_ = 4 six atoms, B site, two angle types
    w41 = np.dot(R_phi(theta_), np.array([5 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
    w42 = np.dot(R_phi(theta_), np.array([5 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
    w43 = np.dot(R_phi(theta_), np.array([-1 / 2, -3 * rt3 / 2, d]).reshape(3, 1)) * a_
    w44 = np.dot(R_phi(theta_), np.array([-2, -rt3, d]).reshape(3, 1)) * a_
    w45 = np.dot(R_phi(theta_), np.array([-2, rt3, d]).reshape(3, 1)) * a_
    w46 = np.dot(R_phi(theta_), np.array([-1 / 2, 3 * rt3 / 2, d]).reshape(3, 1)) * a_
    FouA_up = [w41, w42, w43, w44, w45, w46]
    FouB_up = list(-1 * np.array(FouA_up))

    KAB1, KBA1 = kA[:3], kB[:3]
    KAA2, KBB2 = kA[3:9], kB[3:9]
    KAB3, KBA3 = kA[9:12], kB[9:12]
    KAB4, KBA4 = kA[12:18], kB[12:18]

    DAAf, DBBf, DBAf, DABf = 0, 0, 0, 0  # f = first  左上角2*2*3
    for im in range(3):
        DAAf = DAAf + \
               KAA2[im] * np.exp(1j * np.matmul(v_, SecA[im])) + \
               KAA2[im + 3] * np.exp(1j * np.matmul(v_, SecA[im + 3]))
        DBBf = DBBf + \
               KBB2[im] * np.exp(1j * np.matmul(v_, SecB[im])) + \
               KBB2[im + 3] * np.exp(1j * np.matmul(v_, SecB[im + 3]))
        DABf = DABf + \
               KAB1[im] * np.exp(1j * np.matmul(v_, FirA[im])) + \
               KAB3[im] * np.exp(1j * np.matmul(v_, ThiA[im])) + \
               KAB4[im] * np.exp(1j * np.matmul(v_, FouA[im])) + \
               KAB4[im + 3] * np.exp(1j * np.matmul(v_, FouA[im + 3]))
        DBAf = DBAf + \
               KBA1[im] * np.exp(1j * np.matmul(v_, FirB[im])) + \
               KBA3[im] * np.exp(1j * np.matmul(v_, ThiB[im])) + \
               KBA4[im] * np.exp(1j * np.matmul(v_, FouB[im])) + \
               KBA4[im + 3] * np.exp(1j * np.matmul(v_, FouB[im + 3]))

    DAAs, DBBs, DBAs, DABs = 0, 0, 0, 0  # s = second 右下角2*2*3
    for im in range(3):
        DAAs = DAAs + \
               KAA2[im] * np.exp(1j * np.matmul(w_, SecA_up[im])) + \
               KAA2[im + 3] * np.exp(1j * np.matmul(w_, SecA_up[im + 3]))
        DBBs = DBBs + \
               KBB2[im] * np.exp(1j * np.matmul(w_, SecB_up[im])) + \
               KBB2[im + 3] * np.exp(1j * np.matmul(w_, SecB_up[im + 3]))
        DABs = DABs + \
               KAB1[im] * np.exp(1j * np.matmul(w_, FirA_up[im])) + \
               KAB3[im] * np.exp(1j * np.matmul(w_, ThiA_up[im])) + \
               KAB4[im] * np.exp(1j * np.matmul(w_, FouA_up[im])) + \
               KAB4[im + 3] * np.exp(1j * np.matmul(w_, FouA_up[im + 3]))
        DBAs = DBAs + \
               KBA1[im] * np.exp(1j * np.matmul(w_, FirB_up[im])) + \
               KBA3[im] * np.exp(1j * np.matmul(w_, ThiB_up[im])) + \
               KBA4[im] * np.exp(1j * np.matmul(w_, FouB_up[im])) + \
               KBA4[im + 3] * np.exp(1j * np.matmul(w_, FouB_up[im + 3]))

    KAB1_, KBA1_ = kA_[:3], kB_[:3]
    KAA2_, KBB2_ = kA_[3:9], kB_[3:9]
    KAB3_, KBA3_ = kA_[9:12], kB_[9:12]
    KAB4_, KBA4_ = kA_[12:18], kB_[12:18]

    D = np.zeros([superatoms * 3, superatoms * 3], dtype=np.complex)
    # D矩阵每组的数据格式是: AA BB AB BA；
    # 其位置形式如下：np.diag([AA, BB, CC, DD, EE, ..., AA', BB', CC', DD', EE', ...])
    AA = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    BB = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf

    # for index in range(14):  # 构建AB的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0:i0 + 3] = AA  # AA
    #     elif i % 2 != 0:  # B
    #         D[i0:i0 + 3, i0:i0 + 3] = BB  # BB0
    #         D[i0 - 3:i0, i0:i0 + 3] = DABf  # AB
    #         D[i0:i0 + 3, i0 - 3:i0] = np.conj(DABf).T  # BA, -DBAf  # np.conj(DABs.T)
    #     else:  # A
    #         D[i0:i0 + 3, i0:i0 + 3] = AA  # BB
    #         D[i0 - 3:i0, i0:i0 + 3] = DABf  # AB
    #         D[i0:i0 + 3, i0 - 3:i0] = np.conj(DABf).T  # BA

    # for index in range(14):  # 构建AB的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         print([i0, i0 + 3, i0, i0 + 3])
    #     elif i % 2 != 0:  # B
    #         print([i0, i0 + 3, i0, i0 + 3])
    #         print([i0 - 3, i0, i0, i0 + 3])
    #         print([i0, i0 + 3, i0 - 3, i0])
    #     else:  # A
    #         print([i0, i0 + 3, i0, i0 + 3])
    #         print([i0 - 3, i0, i0, i0 + 3])
    #         print([i0, i0 + 3, i0 - 3, i0])
    #     print()

    # for index in range(12):  # 构建AB主对角线-2的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -AA  # AC  == AA
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(AA.T)  # CA  == AA
    #     elif i % 2 == 0:
    #         D[i0 - 3:i0, i0 + 6:i0 + 9] = -DABf  # AD
    #         D[i0 + 6:i0 + 9, i0 - 3:i0] = -np.conj(DABf.T)  # DA
    #
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -BB  # BD
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(BB.T)  # DB
    #     else:
    #         D[i0 - 3:i0, i0 + 6:i0 + 9] = -DBAf  # AD
    #         D[i0 + 6:i0 + 9, i0 - 3:i0] = -np.conj(DBAf.T)  # DA
    #
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -AA  # BD
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(AA.T)  # DB

    # for index in range(14, 28):  # 构建AB的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0:i0 + 3] = -AA  # AA
    #     elif i % 2 != 0:  # B
    #         D[i0:i0 + 3, i0:i0 + 3] = -BB  # BB0
    #         D[i0 - 3:i0, i0:i0 + 3] = -DABf  # AB
    #         D[i0:i0 + 3, i0 - 3:i0] = -np.conj(DABf.T)  # BA, -DBAf  # np.conj(DABs.T)
    #     else:  # A
    #         D[i0:i0 + 3, i0:i0 + 3] = -AA  # BB
    #         D[i0 - 3:i0, i0:i0 + 3] = -DABf  # AB
    #         D[i0:i0 + 3, i0 - 3:i0] = -np.conj(DABf.T)  # BA
    # for index in range(12, 24):  # 构建AB主对角线-2的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -AA  # AC  == AA
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(AA.T)  # CA  == AA
    #     elif i % 2 == 0:
    #         D[i0 - 3:i0, i0 + 6:i0 + 9] = -DABf  # AD
    #         D[i0 + 6:i0 + 9, i0 - 3:i0] = -np.conj(DABf.T)  # DA
    #
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -BB  # BD
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(BB.T)  # DB
    #     else:
    #         D[i0 - 3:i0, i0 + 6:i0 + 9] = -DBAf  # AD
    #         D[i0 + 6:i0 + 9, i0 - 3:i0] = -np.conj(DBAf.T)  # DA
    #
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = -AA  # BD
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(AA.T)  # DB
    # for index in range(14):  # 构建A'B'的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0:i0 + 3] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    #     else:
    #         D[i0:i0 + 3, i0:i0 + 3] = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf
    #         D[i0 - 3:i0, i0:i0 + 3] = -DABf
    #         D[i0:i0 + 3, i0 - 3:i0] = -np.conj(DABs.T)  # -DBAf  # np.conj(DABs.T)
    # for index in range(12):  # 构建AB主对角线-2的矩阵
    #     i0 = index * 3
    #     if index == 0:
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    #     else:
    #         D[i0:i0 + 3, i0 + 6:i0 + 9] = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf
    #         D[i0 - 3:i0, i0 + 6:i0 + 9] = -DABf
    #         D[i0 + 6:i0 + 9, i0:i0 + 3] = -np.conj(DABs.T)  # -DBAf  # np.conj(DABs.T)
    #         D[i0 + 6:i0 + 9, i0 - 3:i0] = -np.conj(DABs.T)  # -DBAf  # np.conj(DABs.T)

    # D[0:3, 6:9] = Daaf
    # D[3:6, 9:12] = Dbbf
    # D[0:3, 9:12] = Dabf
    # D[3:6, 6:9] = Dbaf
    #
    # D[6:9, 0:3] = Daas
    # D[9:12, 3:6] = Dbbs
    # D[6:9, 3:6] = Dabs
    # D[9:12, 0:3] = Dbas
    # print(D)
    return D


# def fcm(*args, **kwargs):  # force constant matrix
#     # sp_: sphere
#     theta_ = kwargs['theta']
#     v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)  # v_: vector 0;  w_ : vector w_0.5
#     # w_ = np.c_[kwargs['v'].reshape(1, 2), d*a].reshape(1, 3)
#     w_ = v_
#     kA = []
#     kB = []
#     kA_ = []  # kA is A' atom
#     kB_ = []
#     rt3 = 3**0.5
#     # 在这里，我们认为也要对13年文献的初始force constant转到相对初始的张量状态
#     # 旋转矩阵的逆矩阵等于旋转矩阵的转置   inv.(U) == U.T
#     Phi_sp_1 = force_constant[0:3, 0:3]  # 层1不需要转, 或者说转Phi(0)， 严格按照13年FIG2的顺序对应关系
#     Phi_sp_2 = R_phi(np.pi/2) @ force_constant[3:6, 3:6] @ R_phi(np.pi/2).T  # 层2转 π/2
#     Phi_sp_3 = R_phi(np.pi) @ force_constant[6:9, 6:9] @ R_phi(np.pi).T  # 层3转 π
#     Phi_sp_4_up = R_phi(-np.arctan(rt3/5)) @ force_constant[9:12, 9:12] @ R_phi(-np.arctan(rt3/5)).T
#     # 层4上面转 arctan(rt3/5)
#     Phi_sp_4_down = R_phi(-2*np.pi+np.arctan(rt3 / 5)) @ force_constant[9:12, 9:12] @ R_phi(-2*np.pi+np.arctan(rt3 / 5)).T
#     # 层4下面转 2*π-arctan(rt3/5)
#     # The force constant matrices for all atoms from four neighbor atomic spheres can be found from Phi^(n) using the
#     # rotations around the Z axis by an angle ϕ in a clockwise direction.
#     sigma_y = np.diag([1, -1, 1])  # the reflections in the XZ  文献中给出的sigma矩阵
#     a_ = a
#
#     # 来自旋转角度theta的作用部分
#     Phi_up_1 = R_phi(theta_) @ Phi_sp_1 @ R_phi(theta_).T
#     Phi_up_2 = R_phi(theta_) @ Phi_sp_2 @ R_phi(theta_).T
#     Phi_up_3 = R_phi(theta_) @ Phi_sp_3 @ R_phi(theta_).T
#     Phi_up_4_u = R_phi(theta_) @ Phi_sp_4_up @ R_phi(theta_).T
#     Phi_up_4_d = R_phi(theta_) @ Phi_sp_4_down @ R_phi(theta_).T
#     # Phi_up_1 = Phi_sp_1
#     # Phi_up_2 = Phi_sp_2
#     # Phi_up_3 = Phi_sp_3
#     # Phi_up_4_u = Phi_sp_4_up
#     # Phi_up_4_d = Phi_sp_4_down
#
#     U = R_phi(np.pi)
#     # 通过旋转矩阵U可以将以A类原子为中心的近邻原子的相应的K矩阵转化成以B类原子为中心的
#     # sp_ = 1 three atoms, B site
#     for im in range(3):  # 转2π/3
#         r_ = R_phi(2 * np.pi / 3 * im) @ Phi_sp_1 @ R_phi(2 * np.pi / 3 * im).T
#         kA.append(r_)
#         kB.append(U.T @ r_ @ U)
#         # 增加旋转层的Phi ij向量
#         ra_ = R_phi(2 * np.pi / 3 * im) @ Phi_up_1 @ R_phi(2 * np.pi / 3 * im).T
#         kA_.append(ra_)
#         kB_.append(U.T @ ra_ @ U)
#
#     # sp_ = 2 six atoms, A site
#     for im in range(6):
#         r_ = R_phi(im * np.pi / 3) @ Phi_sp_2 @ R_phi(im * np.pi / 3).T
#         kA.append(r_)
#         kB.append(U.T @ r_ @ U)
#
#         ra_ = R_phi(im * np.pi / 3) @ Phi_up_2 @ R_phi(im * np.pi / 3).T
#         kA_.append(ra_)
#         kB_.append(U.T @ ra_ @ U)
#     # sp_ = 3 three atoms, B site
#     for im in range(3):  # 2*np.pi/3
#         r_ = R_phi(2 * np.pi / 3 * im) @ Phi_sp_3 @ R_phi(2 * np.pi / 3 * im).T
#         kA.append(r_)
#         kB.append(U.T @ r_ @ U)
#
#         ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_3 @ R_phi(im * 2 * np.pi / 3).T
#         kA_.append(ra_)
#         kB_.append(U.T @ ra_ @ U)
#     # sp_ = 4 six atoms, B site, two angle types
#     for im in range(3):
#         r_ = R_phi(im * 2 * np.pi / 3) @ Phi_sp_4_up @ R_phi(im * 2 * np.pi / 3).T
#         kA.append(r_)
#         kB.append(U.T @ r_ @ U)
#
#         r_ = R_phi(im * 2 * np.pi / 3) @ Phi_sp_4_down @ R_phi(im * 2 * np.pi / 3).T
#         kA.append(r_)
#         kB.append(U.T @ r_ @ U)
#
#         ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_4_u @ R_phi(im * 2 * np.pi / 3).T
#         kA_.append(ra_)
#         kB_.append(U.T @ ra_ @ U)
#
#         ra_ = R_phi(im * 2 * np.pi / 3) @ Phi_up_4_d @ R_phi(im * 2 * np.pi / 3).T
#         kA_.append(ra_)
#         kB_.append(U.T @ ra_ @ U)
#
#     # return value 分为: 0,3; 3,9; 9,12; 12,18
#     v11 = np.array([1, 0, 0]).reshape(3, 1) * a_
#     v12 = np.array([-1 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
#     v13 = np.array([-1 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
#     FirA = [v11, v12, v13]
#     FirB = list(-1 * np.array(FirA))
#     # sp_ = 2 six atoms, A site
#     v21 = np.array([0, rt3, 0]).reshape(3, 1) * a_
#     v22 = np.array([3 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
#     v23 = np.array([3 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
#     v24 = np.array([0, -rt3, 0]).reshape(3, 1) * a_
#     v25 = np.array([-3 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
#     v26 = np.array([-3 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
#     SecA = [v21, v22, v23, v24, v25, v26]
#     SecB = list(-1 * np.array(SecA))
#     # sp_ = 3 three atoms, B site
#     v31 = np.array([-2, 0, 0]).reshape(3, 1) * a_
#     v32 = np.array([1, rt3, 0]).reshape(3, 1) * a_
#     v33 = np.array([1, -rt3, 0]).reshape(3, 1) * a_
#     ThiA = [v31, v32, v33]
#     ThiB = list(-1 * np.array(ThiA))
#     # sp_ = 4 six atoms, B site, two angle types
#     v41 = np.array([5 / 2, rt3 / 2, 0]).reshape(3, 1) * a_
#     v42 = np.array([5 / 2, -rt3 / 2, 0]).reshape(3, 1) * a_
#     v43 = np.array([-1 / 2, -3 * rt3 / 2, 0]).reshape(3, 1) * a_
#     v44 = np.array([-2, -rt3, 0]).reshape(3, 1) * a_
#     v45 = np.array([-2, rt3, 0]).reshape(3, 1) * a_
#     v46 = np.array([-1 / 2, 3 * rt3 / 2, 0]).reshape(3, 1) * a_
#     FouA = [v41, v42, v43, v44, v45, v46]
#     FouB = list(-1 * np.array(FouA))
#
#     # layer 2 vectors
#     w11 = np.dot(R_phi(theta_), np.array([1, 0, d]).reshape(3, 1)) * a_
#     w12 = np.dot(R_phi(theta_), np.array([-1 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
#     w13 = np.dot(R_phi(theta_), np.array([-1 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
#     FirA_up = [w11, w12, w13]
#     FirB_up = list(-1 * np.array(FirA_up))
#     # sp_ = 2 six atoms, A site
#     w21 = np.dot(R_phi(theta_), np.array([0, rt3, d]).reshape(3, 1)) * a_
#     w22 = np.dot(R_phi(theta_), np.array([3 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
#     w23 = np.dot(R_phi(theta_), np.array([3 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
#     w24 = np.dot(R_phi(theta_), np.array([0, -rt3, d]).reshape(3, 1)) * a_
#     w25 = np.dot(R_phi(theta_), np.array([-3 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
#     w26 = np.dot(R_phi(theta_), np.array([-3 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
#     SecA_up = [w21, w22, w23, w24, w25, w26]
#     SecB_up = list(-1 * np.array(SecA_up))
#     # sp_ = 3 three atoms, B site
#     w31 = np.dot(R_phi(theta_), np.array([-2, 0, d]).reshape(3, 1)) * a_
#     w32 = np.dot(R_phi(theta_), np.array([1, rt3, d]).reshape(3, 1)) * a_
#     w33 = np.dot(R_phi(theta_), np.array([1, -rt3, d]).reshape(3, 1)) * a_
#     ThiA_up = [w31, w32, w33]
#     ThiB_up = list(-1 * np.array(ThiA_up))
#     # sp_ = 4 six atoms, B site, two angle types
#     w41 = np.dot(R_phi(theta_), np.array([5 / 2, rt3 / 2, d]).reshape(3, 1)) * a_
#     w42 = np.dot(R_phi(theta_), np.array([5 / 2, -rt3 / 2, d]).reshape(3, 1)) * a_
#     w43 = np.dot(R_phi(theta_), np.array([-1 / 2, -3 * rt3 / 2, d]).reshape(3, 1)) * a_
#     w44 = np.dot(R_phi(theta_), np.array([-2, -rt3, d]).reshape(3, 1)) * a_
#     w45 = np.dot(R_phi(theta_), np.array([-2, rt3, d]).reshape(3, 1)) * a_
#     w46 = np.dot(R_phi(theta_), np.array([-1 / 2, 3 * rt3 / 2, d]).reshape(3, 1)) * a_
#     FouA_up = [w41, w42, w43, w44, w45, w46]
#     FouB_up = list(-1 * np.array(FouA_up))
#
#     KAB1, KBA1 = kA[:3], kB[:3]
#     KAA2, KBB2 = kA[3:9], kB[3:9]
#     KAB3, KBA3 = kA[9:12], kB[9:12]
#     KAB4, KBA4 = kA[12:18], kB[12:18]
#
#     DAAf, DBBf, DBAf, DABf = 0, 0, 0, 0  # f = first  左上角2*2*3
#     for im in range(3):
#         DAAf = DAAf + \
#                KAA2[im] * np.exp(1j * np.matmul(v_, SecA[im])) + \
#                KAA2[im + 3] * np.exp(1j * np.matmul(v_, SecA[im + 3]))
#         DBBf = DBBf + \
#                KBB2[im] * np.exp(1j * np.matmul(v_, SecB[im])) + \
#                KBB2[im + 3] * np.exp(1j * np.matmul(v_, SecB[im + 3]))
#         DABf = DABf + \
#                KAB1[im] * np.exp(1j * np.matmul(v_, FirA[im])) + \
#                KAB3[im] * np.exp(1j * np.matmul(v_, ThiA[im])) + \
#                KAB4[im] * np.exp(1j * np.matmul(v_, FouA[im])) + \
#                KAB4[im + 3] * np.exp(1j * np.matmul(v_, FouA[im + 3]))
#         DBAf = DBAf + \
#                KBA1[im] * np.exp(1j * np.matmul(v_, FirB[im])) + \
#                KBA3[im] * np.exp(1j * np.matmul(v_, ThiB[im])) + \
#                KBA4[im] * np.exp(1j * np.matmul(v_, FouB[im])) + \
#                KBA4[im + 3] * np.exp(1j * np.matmul(v_, FouB[im + 3]))
#
#     DAAs, DBBs, DBAs, DABs = 0, 0, 0, 0  # s = second 右下角2*2*3
#     for im in range(3):
#         DAAs = DAAs + \
#                KAA2[im] * np.exp(1j * np.matmul(w_, SecA_up[im])) + \
#                KAA2[im + 3] * np.exp(1j * np.matmul(w_, SecA_up[im + 3]))
#         DBBs = DBBs + \
#                KBB2[im] * np.exp(1j * np.matmul(w_, SecB_up[im])) + \
#                KBB2[im + 3] * np.exp(1j * np.matmul(w_, SecB_up[im + 3]))
#         DABs = DABs + \
#                KAB1[im] * np.exp(1j * np.matmul(w_, FirA_up[im])) + \
#                KAB3[im] * np.exp(1j * np.matmul(w_, ThiA_up[im])) + \
#                KAB4[im] * np.exp(1j * np.matmul(w_, FouA_up[im])) + \
#                KAB4[im + 3] * np.exp(1j * np.matmul(w_, FouA_up[im + 3]))
#         DBAs = DBAs + \
#                KBA1[im] * np.exp(1j * np.matmul(w_, FirB_up[im])) + \
#                KBA3[im] * np.exp(1j * np.matmul(w_, ThiB_up[im])) + \
#                KBA4[im] * np.exp(1j * np.matmul(w_, FouB_up[im])) + \
#                KBA4[im + 3] * np.exp(1j * np.matmul(w_, FouB_up[im + 3]))
#
#     KAB1_, KBA1_ = kA_[:3], kB_[:3]
#     KAA2_, KBB2_ = kA_[3:9], kB_[3:9]
#     KAB3_, KBA3_ = kA_[9:12], kB_[9:12]
#     KAB4_, KBA4_ = kA_[12:18], kB_[12:18]
#     Daaf, Dbbf, Dabf, Dbaf = 0, 0, 0, 0  # s = second 右下角2*2*3
#     for rd in range(3):  # 右上角2*2*3
#         # 这里的Phi_rtd代码和projection代码可以合在一起
#         Daaf = Daaf + \
#                np.diag(FCMs(SecA[rd], SecA_up[rd])*np.exp(1j * np.matmul(w_, SecA[rd]-SecA_up[rd]))) + \
#                np.diag(FCMs(SecA[rd + 3], SecA_up[rd + 3])*np.exp(1j * np.matmul(w_, SecA[rd + 3]-SecA_up[rd + 3])))
#         Dbbf = Dbbf + \
#                np.diag(FCMs(SecB[rd], SecB_up[rd])*np.exp(1j * np.matmul(w_, SecB[rd]-SecB_up[rd]))) + \
#                np.diag(FCMs(SecB[rd + 3], SecB_up[rd + 3])*np.exp(1j * np.matmul(w_, SecB[rd + 3]-SecB_up[rd + 3])))
#         Dabf = Dabf + \
#                np.diag(FCMs(FirA[rd], FirA_up[rd])*np.exp(1j * np.matmul(w_, FirA[rd]-FirA_up[rd]))) + \
#                np.diag(FCMs(ThiA[rd], ThiA_up[rd])*np.exp(1j * np.matmul(w_, ThiA[rd]-ThiA_up[rd]))) + \
#                np.diag(FCMs(FouA[rd], FouA_up[rd])*np.exp(1j * np.matmul(w_, FouA[rd]-FouA_up[rd]))) + \
#                np.diag(FCMs(FouA[rd], FouA_up[rd])*np.exp(1j * np.matmul(w_, FouA[rd]-FouA_up[rd])))
#         Dbaf = Dbaf + \
#                np.diag(FCMs(FirB[rd], FirB_up[rd])*np.exp(1j * np.matmul(w_, FirB[rd]-FirB_up[rd]))) + \
#                np.diag(FCMs(ThiB[rd], ThiB_up[rd])*np.exp(1j * np.matmul(w_, ThiB[rd]-ThiB_up[rd]))) + \
#                np.diag(FCMs(FouB[rd], FouB_up[rd])*np.exp(1j * np.matmul(w_, FouB[rd]-FouB_up[rd]))) + \
#                np.diag(FCMs(FouB[rd], FouB_up[rd])*np.exp(1j * np.matmul(w_, FouB[rd]-FouB_up[rd])))
#
#     Daas, Dbbs, Dabs, Dbas = 0, 0, 0, 0  # s = second 右下角2*2*3
#     for rd in range(3):  # 右上角2*2*3
#         # 这里的Phi_rtd代码和projection代码可以合在一起
#         Daas = Daas + \
#                np.diag(FCMs(SecA_up[rd], SecA[rd])*np.exp(1j * np.matmul(w_, SecA_up[rd]-SecA[rd]))) + \
#                np.diag(FCMs(SecA_up[rd + 3], SecA[rd + 3])*np.exp(1j * np.matmul(w_, SecA_up[rd + 3]-SecA[rd + 3])))
#         Dbbs = Dbbs + \
#                np.diag(FCMs(SecB_up[rd], SecB[rd])*np.exp(1j * np.matmul(w_, SecB_up[rd]-SecB[rd]))) + \
#                np.diag(FCMs(SecB_up[rd + 3], SecB[rd + 3])*np.exp(1j * np.matmul(w_, SecB_up[rd + 3]-SecB[rd + 3])))
#         Dabs = Dabs + \
#                np.diag(FCMs(FirA_up[rd], FirA[rd])*np.exp(1j * np.matmul(w_, FirA_up[rd]-FirA[rd]))) + \
#                np.diag(FCMs(ThiA_up[rd], ThiA[rd])*np.exp(1j * np.matmul(w_, ThiA_up[rd]-ThiA[rd]))) + \
#                np.diag(FCMs(FouA_up[rd], FouA[rd])*np.exp(1j * np.matmul(w_, FouA_up[rd]-FouA[rd]))) + \
#                np.diag(FCMs(FouA_up[rd], FouA[rd])*np.exp(1j * np.matmul(w_, FouA_up[rd]-FouA[rd])))
#         Dbas = Dbas + \
#                np.diag(FCMs(FirB_up[rd], FirB[rd])*np.exp(1j * np.matmul(w_, FirB_up[rd]-FirB[rd]))) + \
#                np.diag(FCMs(ThiB_up[rd], ThiB[rd])*np.exp(1j * np.matmul(w_, ThiB_up[rd]-ThiB[rd]))) + \
#                np.diag(FCMs(FouB_up[rd], FouB[rd])*np.exp(1j * np.matmul(w_, FouB_up[rd]-FouB[rd]))) + \
#                np.diag(FCMs(FouB_up[rd], FouB[rd])*np.exp(1j * np.matmul(w_, FouB_up[rd]-FouB[rd])))
#
#     # for rd in range(3):  # 右上角2*2*3
#     #     # 这里的Phi_rtd代码和projection代码可以合在一起
#     #     Daaf = Daaf + \
#     #            np.diag(FCMs(SecA[rd], SecA_up[rd]))*np.exp(1j * np.matmul(w_, -SecA[rd]+SecA_up[rd])) + \
#     #            np.diag(FCMs(SecA[rd + 3], SecA_up[rd + 3]))*np.exp(1j * np.matmul(w_, -SecA[rd + 3]+SecA_up[rd + 3]))
#     #     Dbbf = Dbbf + \
#     #            np.diag(FCMs(SecB[rd], SecB_up[rd]))*np.exp(1j * np.matmul(w_, -SecB[rd]+SecB_up[rd])) + \
#     #            np.diag(FCMs(SecB[rd + 3], SecB_up[rd + 3]))*np.exp(1j * np.matmul(w_, -SecB[rd + 3]+SecB_up[rd + 3]))
#     #     Dabf = Dabf + \
#     #            np.diag(FCMs(FirA[rd], FirA_up[rd]))*np.exp(1j * np.matmul(w_, -FirA[rd]+FirA_up[rd])) + \
#     #            np.diag(FCMs(ThiA[rd], ThiA_up[rd]))*np.exp(1j * np.matmul(w_, -ThiA[rd]+ThiA_up[rd])) + \
#     #            np.diag(FCMs(FouA[rd], FouA_up[rd]))*np.exp(1j * np.matmul(w_, -FouA[rd]+FouA_up[rd]))+ \
#     #            np.diag(FCMs(FouA[rd], FouA_up[rd]))*np.exp(1j * np.matmul(w_, -FouA[rd]+FouA_up[rd]))
#     #     Dbaf = Dbaf + \
#     #            np.diag(FCMs(FirB[rd], FirB_up[rd]))*np.exp(1j * np.matmul(w_, -FirB[rd]+FirB_up[rd])) + \
#     #            np.diag(FCMs(ThiB[rd], ThiB_up[rd]))*np.exp(1j * np.matmul(w_, -ThiB[rd]+ThiB_up[rd])) + \
#     #            np.diag(FCMs(FouB[rd], FouB_up[rd]))*np.exp(1j * np.matmul(w_, -FouB[rd]+FouB_up[rd])) + \
#     #            np.diag(FCMs(FouB[rd], FouB_up[rd]))*np.exp(1j * np.matmul(w_, -FouB[rd]+FouB_up[rd]))
#     #
#     # Daas, Dbbs, Dabs, Dbas = 0, 0, 0, 0  # s = second 右下角2*2*3
#     # for rd in range(3):  # 右上角2*2*3
#     #     # 这里的Phi_rtd代码和projection代码可以合在一起
#     #     Daas = Daas + \
#     #            np.diag(FCMs(SecA_up[rd], SecA[rd]))*np.exp(1j * np.matmul(w_, -SecA_up[rd]+SecA[rd])) + \
#     #            np.diag(FCMs(SecA_up[rd + 3], SecA[rd + 3]))*np.exp(1j * np.matmul(w_, -SecA_up[rd + 3]++SecA[rd + 3]))
#     #     Dbbs = Dbbs + \
#     #            np.diag(FCMs(SecB_up[rd], SecB[rd]))*np.exp(1j * np.matmul(w_, -SecB_up[rd]+SecB[rd])) + \
#     #            np.diag(FCMs(SecB_up[rd + 3], SecB[rd + 3]))*np.exp(1j * np.matmul(w_, --SecB_up[rd + 3]+SecB[rd + 3]))
#     #     Dabs = Dabs + \
#     #            np.diag(FCMs(FirA_up[rd], FirA[rd]))*np.exp(1j * np.matmul(w_, -FirA_up[rd]+FirA[rd])) + \
#     #            np.diag(FCMs(ThiA_up[rd], ThiA[rd]))*np.exp(1j * np.matmul(w_, -ThiA_up[rd]+ThiA[rd])) + \
#     #            np.diag(FCMs(FouA_up[rd], FouA[rd]))*np.exp(1j * np.matmul(w_, -FouA_up[rd]+FouA[rd])) + \
#     #            np.diag(FCMs(FouA_up[rd], FouA[rd]))*np.exp(1j * np.matmul(w_, -FouA_up[rd]+FouA[rd]))
#     #     Dbas = Dbas + \
#     #            np.diag(FCMs(FirB_up[rd], FirB[rd]))*np.exp(1j * np.matmul(w_, -FirB_up[rd]+FirB[rd])) + \
#     #            np.diag(FCMs(ThiB_up[rd], ThiB[rd]))*np.exp(1j * np.matmul(w_, -ThiB_up[rd]+ThiB[rd])) + \
#     #            np.diag(FCMs(FouB_up[rd], FouB[rd]))*np.exp(1j * np.matmul(w_, -FouB_up[rd]+FouB[rd])) + \
#     #            np.diag(FCMs(FouB_up[rd], FouB[rd]))*np.exp(1j * np.matmul(w_, -FouB_up[rd]+FouB[rd]))
#
#     D = np.zeros([4 * 3, 4 * 3], dtype=np.complex)
#     # D矩阵每组的数据格式是: AA BB AB BA；
#     # 其位置形式如下：
#     # AA AB
#     # BA BB
#     D[0:3, 3:6] = -DABf
#     D[3:6, 0:3] = -DBAf  # np.conj(DABs.T)
#     D[0:3, 0:3] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
#     D[3:6, 3:6] = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf

#
#     D[6:9, 9:12] = -DABs
#     D[9:12, 6:9] = -DBAs  # np.conj(DABs.T)
#     D[6:9, 6:9] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAs
#     D[9:12, 9:12] = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBs
#
#     D[0:3, 6:9] = Daaf
#     D[3:6, 9:12] = Dbbf
#     D[0:3, 9:12] = Dabf
#     D[3:6, 6:9] = Dbaf
#
#     D[6:9, 0:3] = Daas
#     D[9:12, 3:6] = Dbbs
#     D[6:9, 3:6] = Dabs
#     D[9:12, 0:3] = Dbas
#
#     return D


if __name__ == '__main__':
    # start here
    time1 = time.time()
    # a = 1.42e-10
    # m = 1.99e-26
    a = 0.142  # nm
    d = 6.7 / a  # nm
    m = 1.99
    superatoms = 2
    n = 100

    angle = 0

    # force_constant = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])
    force_constant = np.diag([398.7, 172.8, 98.9,
                              72.9, -46.1, -8.2,
                              -26.4, 33.1, 5.8,
                              1.0, 7.9, -5.2])

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, superatoms * 3])  # 解的矩阵

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
        k = np.array([kx, ky])  # 得到k值，带入D矩阵
        dm = fcm(theta=np.deg2rad(angle), v=k)
        w, t = np.linalg.eig(dm)
        w = list(w)
        w.sort()
        result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存
    s = 3 ** 0.5
    xk = [0, s, s + 1, s + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(3 ** 0.5 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(kk, result)
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    # plt.xlim(0, s + 1)
    # plt.ylim(0, 1)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(s, color='gray', linestyle='--')
    plt.axvline(s + 1, color='gray', linestyle='--')
    plt.title('%.2f ° ' % angle)
    # plt.savefig('png/声子色散.png', dpi=200)
    plt.show()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
