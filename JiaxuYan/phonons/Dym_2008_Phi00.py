#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Dym_2008_Phi00.py    
@Time    :   2023/3/4 21:06  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    听取果阳建议，将2008的源程序加上φ00，因为力的作用是相互的。
    另外注意！需要假设：两层虽然不一样的堆叠方式和结构，但是sum()的力是相同的。
    具体的φ00求解过程参照网站98年例子，或参考代码见"动力学计算声子谱.py"

@version : 220304.1
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import JiaxuYan.phonons.constant_file as cf


def R_phi(theta_):
    return np.array([[np.cos(theta_), np.sin(theta_), 0],
                     [-np.sin(theta_), np.cos(theta_), 0],
                     [0, 0, 1]])


def get_phi00(*args, **kwargs):

    theta_ = kwargs['theta']
    # v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)  # v_: vector 0;  w_ : vector w_0.5
    # w_ = np.c_[kwargs['v'].reshape(1, 2), d*a].reshape(1, 3)
    v_ = kwargs['v']
    w_ = v_
    kA = []
    kB = []
    kA_ = []  # kA is A' atom
    kB_ = []

    Phi_sp_1 = np.diag(fcs[0])  # 层1不需要转, 或者说转Phi(0)， 严格按照13年FIG2的顺序对应关系
    Phi_sp_2 = R_phi(np.pi / 2) @ np.diag(fcs[1]) @ R_phi(np.pi / 2).T  # 层2转 π/2
    Phi_sp_3 = R_phi(np.pi) @ np.diag(fcs[2]) @ R_phi(np.pi).T  # 层3转 π
    Phi_sp_4_up = R_phi(-np.arctan(rt3 / 5)) @ np.diag(fcs[3]) @ R_phi(-np.arctan(rt3 / 5)).T
    # 层4上面转 arctan(rt3/5)
    Phi_sp_4_down = R_phi(-2 * np.pi + np.arctan(rt3 / 5)) @ np.diag(fcs[3]) @ R_phi(
        -2 * np.pi + np.arctan(rt3 / 5)).T
    # 层4下面转 2*π-arctan(rt3/5)
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

    d = 6.7 / a  # nm
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

    AA = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)])  # - DAAf
    BB = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)])  # - DBBf

    return AA, BB


def dym08_phi00(*args, **kwargs):
    fcs = np.array(cf.force_constant_98)
    a_ = a
    v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)[0, :]  # [qx, qy, qz]
    theta_ = kwargs['theta']
    # v_ = [2 * np.pi / 3 / a_cc, 2 * np.pi / 3 / rt3 / a_cc]
    f1xxaa = 3 * (fcs[0][0] + fcs[0][1]) / 2  # 1 代表n=1圈
    f1zzaa = 3 * fcs[0][2]
    f1aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f1aa[0, 0] = f1xxaa
    f1aa[1, 1] = f1xxaa
    f1aa[2, 2] = f1zzaa
    # >>>>>>>>分割线<<<<<<<<==
    f1xxab = -1 * (fcs[0][0] * np.exp(1j * v_[0] * a_ / rt3) + 1 / 2 * (fcs[0][0] + 3 * fcs[0][1]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * a_ / 2 / rt3))
    f1yyab = -1 * (fcs[0][1] * np.exp(1j * v_[0] * a_ / rt3) + 1 / 2 * (fcs[0][1] + 3 * fcs[0][0]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * a_ / 2 / rt3))
    f1zzab = -1 * (fcs[0][2] * np.exp(1j * v_[0] * a_ / rt3) + 1 / 2 * (fcs[0][2] + 3 * fcs[0][2]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * a_ / 2 / rt3))
    f1xyab = 1j * rt3 / 2 * (fcs[0][0] - fcs[0][1]) * np.sin(v_[1] / 2 * a_) * np.exp(-1j * v_[0] * a_ / 2 / rt3)
    f1ab = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f1ab[0, 0] = f1xxab
    f1ab[1, 1] = f1yyab
    f1ab[2, 2] = f1zzab
    f1ab[0, 1] = f1xyab
    f1ab[1, 0] = f1xyab  # np.conj(f1xyab).T
    # f2  # >>>>>>>>分割线<<<<<<<<
    f2xxaa = 1 * ((fcs[1][1] + 3 * fcs[1][0]) * (
            np.sin(rt3 * v_[0] + v_[1]) ** 2 * a_ / 4 + np.sin(-rt3 * v_[0] + v_[1]) ** 2 * a_ / 4) + 4 * fcs[1][
                      1] * np.sin(v_[1] * a_ / 2) ** 2)
    f2yyaa = 1 * ((fcs[1][0] + 3 * fcs[1][1]) * (
            np.sin(rt3 * v_[0] + v_[1]) ** 2 * a_ / 4 + np.sin(-rt3 * v_[0] + v_[1]) ** 2 * a_ / 4) + 4 * fcs[1][
                      0] * np.sin(v_[1] * a_ / 2) ** 2)
    f2zzaa = 1 * ((fcs[1][2] + 3 * fcs[1][2]) * (
            np.sin(rt3 * v_[0] + v_[1]) ** 2 * a_ / 4 + np.sin(-rt3 * v_[0] + v_[1]) ** 2 * a_ / 4) + 4 * fcs[1][
                      2] * np.sin(v_[1] * a_ / 2) ** 2)
    f2xyaa = -rt3 * (fcs[1][1] - fcs[1][0]) * (
            np.sin(rt3 * v_[0] + v_[1]) ** 2 * a_ / 4 - np.sin(-rt3 * v_[0] + v_[1]) ** 2 * a_ / 4)
    f2aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f2aa[0, 0] = f2xxaa
    f2aa[1, 1] = f2yyaa
    f2aa[2, 2] = f2zzaa
    f2aa[0, 1] = f2xyaa
    f2aa[1, 0] = f2xyaa  # np.conj(f2xyaa).T
    f2ab = np.zeros((3, 3), dtype=complex)  # 全都是0
    # >>>>>>>>分割线<<<<<<<<
    # F3 = F1 replace f1r f1i f1o by f3r f3i f3o and replace a by -2a
    f3xxaa = 3 * (fcs[2][0] + fcs[2][1]) / 2  # 1 代表n=1圈
    f3zzaa = 3 * fcs[2][2]
    f3aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f3aa[0, 0] = f3xxaa
    f3aa[1, 1] = f3xxaa
    f3aa[2, 2] = f3zzaa
    f3xxab = -1 * (fcs[2][0] * np.exp(1j * v_[0] * (-2 * a_) / rt3) + 1 / 2 * (fcs[2][0] + 3 * fcs[2][1]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * (-2 * a_) / 2 / rt3))
    f3yyab = -1 * (fcs[2][1] * np.exp(1j * v_[0] * (-2 * a_) / rt3) + 1 / 2 * (fcs[2][1] + 3 * fcs[2][0]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * (-2 * a_) / 2 / rt3))
    f3zzab = -1 * (fcs[2][2] * np.exp(1j * v_[0] * (-2 * a_) / rt3) + 1 / 2 * (fcs[2][2] + 3 * fcs[2][2]) * np.cos(
        v_[1] * a_ / 2) * np.exp(-1j * v_[0] * (-2 * a_) / 2 / rt3))
    f3xyab = 1j * rt3 / 2 * (fcs[2][0] - fcs[2][1]) * np.sin(v_[1] / 2 * a_) * np.exp(
        -1j * v_[0] * (-2 * a_) / 2 / rt3)
    f3ab = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f3ab[0, 0] = f3xxab
    f3ab[1, 1] = f3yyab
    f3ab[2, 2] = f3zzab
    f3ab[0, 1] = f3xyab
    f3ab[1, 0] = f3xyab  # np.conj(f3xyab).T
    # >>>>>>>>分割线<<<<<<<<
    # F4AA = F1AA replace f1r f1i f1o by 2f4r 2f4i 2f4o
    th1 = np.arctan(rt3 / 5)
    th2 = 2 * np.pi / 3 - th1
    th3 = 2 * np.pi / 3 + th1
    f4xxaa = 2 * 3 * (fcs[3][0] + fcs[3][1]) / 2  # 1 代表n=1圈
    f4zzaa = 2 * 3 * fcs[3][2]
    f4aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f4aa[0, 0] = f4xxaa
    f4aa[1, 1] = f4xxaa
    f4aa[2, 2] = f4zzaa

    f4xxab = -2 * ((fcs[3][0] * np.cos(th1) ** 2 + fcs[3][1] * np.sin(th1) ** 2) * np.exp(
        1j * v_[0] * 5 * a_ / 2 / rt3) * np.cos(v_[1] * a_ / 2) +
                   (fcs[3][0] * np.cos(th2) ** 2 + fcs[3][1] * np.sin(th2) ** 2) * np.exp(
                -1j * v_[0] * a_ / 2 / rt3) * np.cos(3 * v_[1] * a_ / 2) +
                   (fcs[3][0] * np.cos(th3) ** 2 + fcs[3][1] * np.sin(th3) ** 2) * np.exp(
                -1j * v_[0] * a_ * 2 / rt3) * np.cos(v_[1] * a_))

    f4yyab = -2 * ((fcs[3][1] * np.cos(th1) ** 2 + fcs[3][0] * np.sin(th1) ** 2) * np.exp(
        1j * v_[0] * 5 * a_ / 2 / rt3) * np.cos(v_[1] * a_ / 2) +
                   (fcs[3][1] * np.cos(th2) ** 2 + fcs[3][0] * np.sin(th2) ** 2) * np.exp(
                -1j * v_[0] * a_ / 2 / rt3) * np.cos(3 * v_[1] * a_ / 2) +
                   (fcs[3][1] * np.cos(th3) ** 2 + fcs[3][0] * np.sin(th3) ** 2) * np.exp(
                -1j * v_[0] * a_ * 2 / rt3) * np.cos(v_[1] * a_))
    f4zzab = -2 * ((fcs[3][2] * np.cos(th1) ** 2 + fcs[3][2] * np.sin(th1) ** 2) * np.exp(
        1j * v_[0] * 5 * a_ / 2 / rt3) * np.cos(v_[1] * a_ / 2) +
                   (fcs[3][2] * np.cos(th2) ** 2 + fcs[3][2] * np.sin(th2) ** 2) * np.exp(
                -1j * v_[0] * a_ / 2 / rt3) * np.cos(3 * v_[1] * a_ / 2) +
                   (fcs[3][2] * np.cos(th3) ** 2 + fcs[3][2] * np.sin(th3) ** 2) * np.exp(
                -1j * v_[0] * a_ * 2 / rt3) * np.cos(v_[1] * a_))
    # >>>>>>>>分割线<<<<<<<<
    f4xyab = -1j * (fcs[3][0] - fcs[3][1]) * (
            np.sin(2 * th1) * np.exp(1j * v_[0] * 5 * a_ / 2 / rt3) * np.sin(v_[1] * a_ / 2) +
            np.sin(2 * th2) * np.exp(-1j * v_[0] * a_ / 2 / rt3) * np.sin(3 * v_[1] * a_ / 2) +
            np.sin(2 * th3) * np.exp(-1j * v_[0] * 2 * a_ / rt3) * np.sin(v_[1] * a_))
    f4aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f4aa[0, 0] = f4xxaa
    f4aa[1, 1] = f4xxaa
    f4aa[2, 2] = f4zzaa
    f4ab = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f4ab[0, 0] = f4xxab
    f4ab[1, 1] = f4yyab
    f4ab[2, 2] = f4zzab
    f4ab[0, 1] = f4xyab
    f4ab[1, 0] = f4xyab  # np.conj(f4xyab).T
    # # # f5 # >>>>>>>>分割线<<<<<<<<
    # f5xxaa = 1 * ((fcs[4][0] + 3 * fcs[4][1]) * (
    #         np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
    #             rt3 * a_) / 4) + 4 *
    #               fcs[4][
    #                   0] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    # f5yyaa = 1 * ((fcs[4][1] + 3 * fcs[4][0]) * (
    #         np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
    #             rt3 * a_) / 4) + 4 *
    #               fcs[4][
    #                   0] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    # f5zzaa = 1 * ((fcs[4][2] + 3 * fcs[4][2]) * (
    #         np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
    #             rt3 * a_) / 4) + 4 *
    #               fcs[4][
    #                   2] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    # f5xyaa = -rt3 * (fcs[4][0] - fcs[4][1]) * (
    #         np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 - np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4)
    # f5aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    # f5aa[0, 0] = f5xxaa
    # f5aa[1, 1] = f5yyaa
    # f5aa[2, 2] = f5zzaa
    # f5aa[0, 1] = f5xyaa
    # f5aa[1, 0] = f5xyaa  # np.conj(f5xyaa).T
    # f5ab = f2ab  # 其他都是0
    # fijAA = f1aa + f2aa + f3aa + f4aa + f5aa
    # fijAB = f1ab + f2ab + f3ab + f4ab + f5ab

    # # 在此处插入Phi00的数据。
    sumAA, sumBB = get_phi00(theta=theta_, v=v_)
    fijAA = f1aa + f2aa + f3aa + f4aa
    fijAB = f1ab + f2ab + f3ab + f4ab

    fijBB = fijAA
    fijBA = np.conj(fijAB).T

    Dm = np.zeros((2 * 3, 2 * 3), dtype=complex)
    Dm[0:3, 0:3] = fijAA
    Dm[3:6, 3:6] = fijBB
    Dm[0:3, 3:6] = fijAB
    Dm[3:6, 0:3] = fijBA
    return Dm


if __name__ == '__main__':
    # start here
    rt3 = cf.rt3
    a_cc = 1.42
    a = a_cc * rt3
    M = 12.02 * 1.66 * 1e-24  # g
    m = 12
    n = 200
    angle = 0

    fcs = np.array(cf.force_constant_08)

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 2 * 3])  # 解的矩阵

    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值 (0,1700) (0,rt3)
            kx = i * 2 * np.pi / 3 / a_cc / (n * int(10 * 3 ** 0.5))
            ky = 0
        elif i < (10 + int(10 * 3 ** 0.5)) * n:  # print('2 >> ', i)  # (1700,2700) (rt3,rt3+1)
            kx = 2 * np.pi / 3 / a_cc
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a_cc / 3 ** 0.5
        else:  # print('3 >> ', i)  # (2700,4700) (rt3+1,rt3+3)
            kx = 2 * np.pi / 3 / a_cc - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a_cc
            ky = kx / 3 ** 0.5
            # plt.scatter(kx, ky)
        k = np.array([kx, ky])  # 得到k值，带入D矩阵
        dm = dym08_phi00(theta=np.deg2rad(angle), v=k)
        w, t = np.linalg.eig(dm)
        w = list(w)
        w.sort()
        result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存
    xk = [0, rt3, rt3 + 1, rt3 + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(3 ** 0.5 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(kk, result, lw=2)
    plt.xticks(xk, ["Γ", "K", "M", "Γ"])
    # plt.xlim(0, s + 1)
    # plt.ylim(0, 15)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(xk[1], color='gray', linestyle='--')
    plt.axvline(xk[2], color='gray', linestyle='--')
    plt.title('%.2f ° ' % angle)
    # plt.savefig('png/声子色散.png', dpi=200)
    plt.show()
    print('finished')