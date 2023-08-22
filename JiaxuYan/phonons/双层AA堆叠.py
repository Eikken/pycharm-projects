#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   双层AA堆叠.py    
@Time    :   2023/2/20 18:18  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
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


def FCMs(b_):  # R(θ) is the distance of two atoms
    # 新的层间力常数构建方法
    A_ = 573.76
    B_ = 0.05
    # 13年
    epsilon = 4.6  # meV
    sigma = 0.3276  # nm
    b_ = b_[:, 0]
    rtheta = b_  # rtheta = rj - ri
    ralpha = b_[0]
    rbeta = b_[1]
    norm_r = np.linalg.norm(b_)
    delta_r = A_ * np.exp(-b_ / B_)
    # delta_r = 4*epsilon*(156 * sigma ** 12/norm_r ** 14 - 42*sigma ** 6 / norm_r ** 8)
    # *np.abs(ralpha*rbeta)/norm_r ** 2
    return -delta_r * ralpha * rbeta / norm_r ** 2


def Vr(b_):
    b_ = b_[:, 0]
    # r_ = b_  # rtheta = rj - ri
    # norm_r = np.linalg.norm(r_)
    # A_ = 24.1 * 1e3  # eV Å12
    # B_ = 15.2  # eV Å6
    A_ = 573.76
    B_ = 0.05
    vr = A_ * np.exp(-b_ / B_)
    return vr


def fcm(*args, **kwargs):  # force constant matrix
    # sp_: sphere
    theta_ = kwargs['theta']
    v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)  # v_: vector 0;  w_ : vector w_0.5
    # w_ = np.c_[kwargs['v'].reshape(1, 2), d*a].reshape(1, 3)
    w_ = v_[0, :]  # [0, 0, 0]
    kA = []
    kB = []
    kA_ = []  # kA is A' atom
    kB_ = []
    rt3 = 3 ** 0.5
    Phi_sp_1 = force_constant[0:3, 0:3]  # 层1不需要转, 或者说转Phi(0)， 严格按照13年FIG2的顺序对应关系
    Phi_sp_2 = R_phi(-np.pi / 2) @ force_constant[3:6, 3:6] @ R_phi(-np.pi / 2).T  # 层2转 π/2
    Phi_sp_3 = R_phi(np.pi) @ force_constant[6:9, 6:9] @ R_phi(np.pi).T  # 层3转 π
    Phi_sp_4_up = R_phi(-np.arctan(rt3 / 5)) @ force_constant[9:12, 9:12] @ R_phi(-np.arctan(rt3 / 5)).T
    # 层4上面转 arctan(rt3/5)
    Phi_sp_4_down = R_phi(np.arctan(rt3 / 5)) @ force_constant[9:12, 9:12] @ R_phi(
        np.arctan(rt3 / 5)).T
    a_ = a
    # 来自旋转角度theta的作用部分
    Phi_up_1 = R_phi(theta_) @ Phi_sp_1 @ R_phi(theta_).T
    Phi_up_2 = R_phi(theta_) @ Phi_sp_2 @ R_phi(theta_).T
    Phi_up_3 = R_phi(theta_) @ Phi_sp_3 @ R_phi(theta_).T
    Phi_up_4_u = R_phi(theta_) @ Phi_sp_4_up @ R_phi(theta_).T
    Phi_up_4_d = R_phi(theta_) @ Phi_sp_4_down @ R_phi(theta_).T

    # #################################################################################################
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
    # #################################################################################################
    d_ = 0  # rt3*d/2
    # return value 分为: 0,3; 3,9; 9,12; 12,18
    v11 = np.array([1, 0, d_]).reshape(3, 1) * a_
    v12 = np.array([-1 / 2, -rt3 / 2, d_]).reshape(3, 1) * a_
    v13 = np.array([-1 / 2, rt3 / 2, d_]).reshape(3, 1) * a_
    FirA = [v11, v12, v13]
    # FirA = [vv/rt3 for vv in FirA]
    FirB = list(-1 * np.array(FirA))
    # sp_ = 2 six atoms, A site
    v21 = np.array([0, rt3, d_]).reshape(3, 1) * a_
    v22 = np.array([3 / 2, rt3 / 2, d_]).reshape(3, 1) * a_
    v23 = np.array([3 / 2, -rt3 / 2, d_]).reshape(3, 1) * a_
    v24 = np.array([0, -rt3, d_]).reshape(3, 1) * a_
    v25 = np.array([-3 / 2, -rt3 / 2, d_]).reshape(3, 1) * a_
    v26 = np.array([-3 / 2, rt3 / 2, d_]).reshape(3, 1) * a_
    SecA = [v21, v22, v23, v24, v25, v26]
    # SecA = [vv/rt3 for vv in SecA]
    SecB = list(-1 * np.array(SecA))
    # sp_ = 3 three atoms, B site
    v31 = np.array([-2, 0, d_]).reshape(3, 1) * a_
    v32 = np.array([1, rt3, d_]).reshape(3, 1) * a_
    v33 = np.array([1, -rt3, d_]).reshape(3, 1) * a_
    ThiA = [v31, v32, v33]
    # ThiA = [vv/rt3 for vv in ThiA]
    ThiB = list(-1 * np.array(ThiA))
    # sp_ = 4 six atoms, B site, two angle types
    v41 = np.array([5 / 2, rt3 / 2, d_]).reshape(3, 1) * a_
    v42 = np.array([5 / 2, -rt3 / 2, d_]).reshape(3, 1) * a_
    v43 = np.array([-1 / 2, -3 * rt3 / 2, d_]).reshape(3, 1) * a_
    v44 = np.array([-2, -rt3, d_]).reshape(3, 1) * a_
    v45 = np.array([-2, rt3, d_]).reshape(3, 1) * a_
    v46 = np.array([-1 / 2, 3 * rt3 / 2, d_]).reshape(3, 1) * a_
    FouA = [v41, v42, v43, v44, v45, v46]
    # FouA = [vv/rt3 for vv in FouA]
    FouB = list(-1 * np.array(FouA))
    # #################################################################################################
    d_ = d / 2
    # layer 2 vectors
    w11 = np.dot(R_phi(theta_), np.array([1, 0, d_]).reshape(3, 1)) * a_
    w12 = np.dot(R_phi(theta_), np.array([-1 / 2, -rt3 / 2, d_]).reshape(3, 1)) * a_
    w13 = np.dot(R_phi(theta_), np.array([-1 / 2, rt3 / 2, d_]).reshape(3, 1)) * a_
    FirA_up = [w11, w12, w13]
    # FirA_up = [vv+np.array([1, 0, 0]).reshape(3, 1)*a_ for vv in FirA_up]
    FirB_up = list(-1 * np.array(FirA_up))
    # sp_ = 2 six atoms, A site
    w21 = np.dot(R_phi(theta_), np.array([0, rt3, d_]).reshape(3, 1)) * a_
    w22 = np.dot(R_phi(theta_), np.array([3 / 2, rt3 / 2, d_]).reshape(3, 1)) * a_
    w23 = np.dot(R_phi(theta_), np.array([3 / 2, -rt3 / 2, d_]).reshape(3, 1)) * a_
    w24 = np.dot(R_phi(theta_), np.array([0, -rt3, d_]).reshape(3, 1)) * a_
    w25 = np.dot(R_phi(theta_), np.array([-3 / 2, -rt3 / 2, d_]).reshape(3, 1)) * a_
    w26 = np.dot(R_phi(theta_), np.array([-3 / 2, rt3 / 2, d_]).reshape(3, 1)) * a_
    SecA_up = [w21, w22, w23, w24, w25, w26]
    # SecA_up = [vv+np.array([1, 0, 0]).reshape(3, 1)*a_ / rt3 for vv in SecA_up]
    SecB_up = list(-1 * np.array(SecA_up))
    # sp_ = 3 three atoms, B site
    w31 = np.dot(R_phi(theta_), np.array([-2, 0, d_]).reshape(3, 1)) * a_
    w32 = np.dot(R_phi(theta_), np.array([1, rt3, d_]).reshape(3, 1)) * a_
    w33 = np.dot(R_phi(theta_), np.array([1, -rt3, d_]).reshape(3, 1)) * a_
    ThiA_up = [w31, w32, w33]
    # ThiA_up = [vv+np.array([1, 0, 0]).reshape(3, 1)*a_ / rt3 for vv in ThiA_up]
    ThiB_up = list(-1 * np.array(ThiA_up))
    # sp_ = 4 six atoms, B site, two angle types
    w41 = np.dot(R_phi(theta_), np.array([5 / 2, rt3 / 2, d_]).reshape(3, 1)) * a_
    w42 = np.dot(R_phi(theta_), np.array([5 / 2, -rt3 / 2, d_]).reshape(3, 1)) * a_
    w43 = np.dot(R_phi(theta_), np.array([-1 / 2, -3 * rt3 / 2, d_]).reshape(3, 1)) * a_
    w44 = np.dot(R_phi(theta_), np.array([-2, -rt3, d_]).reshape(3, 1)) * a_
    w45 = np.dot(R_phi(theta_), np.array([-2, rt3, d_]).reshape(3, 1)) * a_
    w46 = np.dot(R_phi(theta_), np.array([-1 / 2, 3 * rt3 / 2, d_]).reshape(3, 1)) * a_
    FouA_up = [w41, w42, w43, w44, w45, w46]
    # FouA_up = [vv+np.array([1, 0, 0]).reshape(3, 1)*a_ / rt3 for vv in FouA_up]
    FouB_up = list(-1 * np.array(FouA_up))
    # #################################################################################################
    KAB1, KBA1 = kA[:3], kB[:3]
    KAA2, KBB2 = kA[3:9], kB[3:9]
    KAB3, KBA3 = kA[9:12], kB[9:12]
    KAB4, KBA4 = kA[12:18], kB[12:18]

    [DAAf, DBBf, DBAf, DABf] = [0 for im in range(4)]  # s = second 右下角2*2*3
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

    HAA = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    HBB = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf
    Ri = [FirB, SecB[:3], SecB[3:], ThiB, FouB[:3], FouB[3:]]
    Rj = [FirA_up, SecA_up[:3], SecA_up[3:], ThiA_up, FouA_up[:3], FouA_up[3:]]
    # Dαα Dαβ
    # Dβα Dββ
    # Dab 矩阵是 (2*3)X(2*3) 的矩阵
    # # FCM传参的时候，args的顺序必须是：第一层原子坐标、第二层原子坐标的顺序
    H12AAp = np.zeros((3, 3), dtype=complex)
    hiiAAp0 = FCMs(FirA_up[0]-FirA[0])*np.exp(1j * np.matmul(v_, FirA_up[0]-FirA[0]))
    hiiAAp = 0
    for im in range(6):
        mat = np.matmul(v_, SecA_up[im])
        hiiAAp = hiiAAp + FCMs(SecA_up[im])*np.exp(-1j * mat)
    H12AAp[0, 0] = -hiiAAp - hiiAAp0
    H12AAp[1, 1] = H12AAp[0, 0]
    H12AAp[2, 2] = H12AAp[0, 0]
    H12AAp[0, 1] = hiiAAp
    H12AAp[0, 2] = hiiAAp
    H12AAp[1, 2] = hiiAAp
    H12ABp = np.zeros((3, 3), dtype=complex)
    hiiABp = 0
    for im in range(3):
        mat1 = np.matmul(v_, FirA_up[im])
        mat2 = np.matmul(v_, ThiA_up[im])
        mat3 = np.matmul(v_, FouA_up[im])
        mat4 = np.matmul(v_, FouA_up[im+3])
        hiiABp = hiiABp + \
                 FCMs(FirA_up[im]) * np.exp(-1j * mat1) + \
                 FCMs(ThiA_up[im]) * np.exp(-1j * mat2) + \
                 FCMs(FouA_up[im]) * np.exp(-1j * mat3) + \
                 FCMs(FouA_up[im]) * np.exp(-1j * mat4)
    H12ABp[0, 0] = -hiiABp
    H12ABp[1, 1] = H12ABp[0, 0]
    H12ABp[2, 2] = H12ABp[0, 0]
    H12ABp[0, 1] = -1j*hiiABp
    H12ABp[0, 2] = -1j*hiiABp
    H12ABp[1, 2] = hiiABp
    if np.real(hiiAAp) != 0:
        print(i, hiiAAp)
    # H12_AAp_11 = hiiAAp
    # H12AAp = np.array([H12_AAp_11, hiiAAp, hiiAAp,
    #                    np.conj(hiiAAp).T, hiiAAp, hiiAAp,
    #                    np.conj(hiiAAp).T, np.conj(hiiAAp).T, hiiAAp]).reshape(3, 3)
    # hiiABp = 0
    # for im in range(3):
    #     hiiABp = hiiABp + FCMs(FirA_up[im])*np.exp(1j * np.matmul(v_, FirA_up[im])) + \
    #              FCMs(ThiA_up[im])*np.exp(1j * np.matmul(v_, ThiA_up[im])) + \
    #              FCMs(FouA_up[im]) * np.exp(1j * np.matmul(v_, FouA_up[im])) + \
    #              FCMs(FouA_up[im+3])*np.exp(1j * np.matmul(v_, FouA_up[im+3]))
    # hiiABp = hii  # np.sum(hiiABp[0])
    #
    # H12ABp = np.array([hiiABp, hiiABp, hiiABp,
    #                    np.conj(hiiABp).T, hiiABp, hiiABp,
    #                    np.conj(hiiABp).T, np.conj(hiiABp).T, hiiABp]).reshape(3, 3)
    # # H12ABp = np.sum(hiiABp[0])
    # H12BAp = H12ABp
    # H12BBp = np.conj(H12ABp).T
    #
    # hiiBBp = hii  # FCMs(FirA_up[0] - FirA[0]) * np.exp(1j * np.matmul(v_, FirA_up[0] - FirA[0]))
    # hiiBBp = 1  # np.sum(hiiBBp[0])
    # H12BBp = np.array([hiiBBp, hiiBBp, hiiBBp,
    #                    np.conj(hiiBBp).T, hiiBBp, hiiBBp,
    #                    np.conj(hiiBBp).T, np.conj(hiiBBp).T, hiiBBp]).reshape(3, 3)
    # H12_BBp = H12_AAp  # np.conj(H12_AAp).T  # HBB' = HAB'.T.conj (厄米共轭矩阵)
    # H12_BAp = np.conj(H12_ABp).T  # H12_ABp  # 这里应该必须都是3*3矩阵
    #  hii >> 11[0]  22[1]  33[2]  12[3]  13[4]  23[5]
    # h1_u = [-114.47, -114.47, 2040.41, 0, 0, 0]
    # h1p_u = [63.17, -17.41, 430.87, 0,  -190.05, 0]
    # h2p_u = [2.73,  43.02,  430.87, -34.89, 95.03, -164.59]
    # h1pp_u = [8.95, 7.19, 12.84,  1.52,  4.15,  2.40]
    # h2pp_u = [6.31,  9.83, 12.84, 0, 0, 4.79]
    #
    # H12AAp = np.zeros((3, 3), dtype=complex)
    # for ii in range(3):
    #     H12AAp[ii][ii] = -2*(h1_u[ii]+2*(2*h1pp_u[ii]*np.cos(w_[0]*rt3*a_/2)*np.cos(w_[1]*rt3*a_/2) + h2pp_u[ii]))*np.cos(w_[2]*d_*a_*rt3)
    # H12AAp[0][1] = 8 * h1pp_u[3]*np.sin(w_[0]*3*a_/2)*np.sin(w_[1]*rt3*a_/2)*np.cos(w_[2]*d_*a_*rt3)
    # H12AAp[0][2] = 8 * h1pp_u[4]*np.sin(w_[0]*3*a_/2)*np.cos(w_[1]*rt3*a_/2)*np.sin(w_[2]*d_*a_*rt3)
    # H12AAp[1][2] = 4 * (2*h1pp_u[5]*np.sin(w_[1]*a_/2*rt3)*np.cos(w_[0]*3*a_/2) + h2pp_u[5]*np.sin(w_[1]*a_*rt3))*np.sin(w_[2]*d_*a_*rt3)
    #
    # H12ABp = np.zeros((3, 3), dtype=complex)
    # for ii in range(3):
    #     H12ABp[ii][ii] = -2 * (h1p_u[ii]*np.exp(-1j*w_[0]*a) + 2*h2p_u[ii]*np.cos(w_[1]*a_/2*rt3)*np.exp(1j*w_[0]*a/2))*np.cos(w_[2]*d_*a_*rt3)
    # H12ABp[0][1] = -4j * h2p_u[3]*np.sin(w_[1]*a_/2*rt3)*np.exp(1j*w_[0]*a_/2)*np.cos(w_[2]*d_*a_*rt3)
    # H12ABp[0][2] = -2j * (h1p_u[4]*np.exp(-1j*w_[0]*a) + 2*h2pp_u[4]*np.exp(1j*w_[0]*a/2)*np.cos(w_[1]*a_/2*rt3))*np.sin(w_[2]*d_*a_*rt3)
    # H12ABp[1][2] = 4 * h2pp_u[5]*np.exp(1j*w_[0]*a/2)*np.sin(w_[1]*a_/2*rt3)*np.sin(w_[2]*d_*a_*rt3)
    #
    H12AAp[1][0] = np.conj(H12AAp[0][1]).T
    H12AAp[2][0] = np.conj(H12AAp[0][2]).T
    H12AAp[2][1] = np.conj(H12AAp[1][2]).T
    H12ABp[1][0] = np.conj(H12ABp[0][1]).T
    H12ABp[2][0] = np.conj(H12ABp[0][2]).T
    H12ABp[2][1] = np.conj(H12ABp[1][2]).T
    # H12AAp[1][0] = H12AAp[0][1]
    # H12AAp[2][0] = H12AAp[0][2]
    # H12AAp[2][1] = H12AAp[1][2]
    # H12ABp[1][0] = H12ABp[0][1]
    # H12ABp[2][0] = H12ABp[0][2]
    # H12ABp[2][1] = H12ABp[1][2]

    # H12ABp = 0
    # phi_n_ij = []  # phi(n = 1,2,3)
    # for im in range(3):
    #     rj = [j[im] for j in Rj]
    #     # print(rj)
    #     # nij = []
    #     tmp = 0
    #     for jj in rj:
    #         mat = np.matmul(v_, jj)
    #         tmp += FCMs(jj) * np.exp(-1j * mat)
    #     phi_n_ij.append(tmp)

    H12BBp = np.conj(H12ABp).T
    H12BAp = H12ABp

    Dab = np.zeros([2 * 3, 2 * 3], dtype=np.complex)  # a = alpha, b = beta
    Dab[0:3, 0:3] = H12AAp
    Dab[3:6, 3:6] = H12BBp
    Dab[0:3, 3:6] = H12ABp
    Dab[3:6, 0:3] = H12BAp
    # HAA_ii = 0
    # HBB_ii = 0
    # for ii in range(3):
    #     HAA_ii += 0.5*(H12AAp[ii, ii] + H12ABp[ii, ii])  # ?
    #     HBB_ii += 0.5*(H12BAp[ii, ii] + H12BBp[ii, ii])  # ?
    HAA_ii = -0.5 * (H12AAp + H12ABp)  # ?
    HBB_ii = -0.5 * (H12BAp + H12BBp)  # ?
    # Dαα 矩阵
    H11_AA = HAA + 2*HAA_ii  # 0.5*(HAA_ii + HAB_ii)
    H11_BB = HBB + 2*HBB_ii  # 0.5*(HAB_ii + HAA_ii)
    H11_AB = -DABf
    H11_BA = np.conj(-DABf).T
    # 矩阵 转置共轭=共轭转置
    Daa = np.zeros([2 * 3, 2 * 3], dtype=np.complex)  # a = alpha, b = beta
    Daa[0:3, 0:3] = H11_AA
    Daa[3:6, 3:6] = H11_BB
    Daa[0:3, 3:6] = H11_AB
    Daa[3:6, 0:3] = H11_BA

    # Dββ 矩阵
    H22_ApAp = H11_AA  # HAA + HAA_ii
    H22_BpBp = H11_BB  # AA + HAB_ii
    H22_ApBp = H11_AB  # np.conj(H11_AB).T
    H22_BpAp = np.conj(H22_ApBp).T

    Dbb = np.zeros([2 * 3, 2 * 3], dtype=np.complex)  # a = alpha, b = beta
    Dbb[0:3, 0:3] = H22_ApAp
    Dbb[3:6, 3:6] = H22_BpBp
    Dbb[0:3, 3:6] = H22_ApBp
    Dbb[3:6, 0:3] = H22_BpAp
    # Dbb = np.conj(Daa).T

    D = np.zeros([2*2 * 3, 2*2 * 3], dtype=np.complex)
    D[0:6, 0:6] = Daa
    D[6:12, 6:12] = np.conj(Daa).T  # Dbb
    D[0:6, 6:12] = Dab
    D[6:12, 0:6] = np.conj(Dab).T  # -DBAf  # np.conj(DABs.T)

    return D


if __name__ == '__main__':
    # start here
    time1 = time.time()
    # a = 1.42e-10
    # m = 1.99e-26
    a = 0.142  # nm
    d = 6.7 / a  # nm
    m = 1.99
    M = 12.02 * 1.66 * 1e-24  # g
    superatoms = 4
    n = 100

    angle = 0

    force_constant98 = np.diag([36.50, 24.50, 9.82,
                                8.80, -3.23, -0.40,
                                3.00, -5.25, 0.15,
                                -1.92, 2.29, -0.58])
    force_constant13 = np.diag([398.7, 172.8, 98.9,
                                72.9, -46.1, -8.2,
                                -26.4, 33.1, 5.8,
                                1.0, 7.9, -5.2])
    force_constant08 = np.diag([25.880, 8.420, 6.183,
                                4.037, -3.044, -0.492,
                                -3.016, 3.948, 0.516,
                                0.564, 0.129, -0.521,
                                1.035, 0.166, 0.110])
    force_constant = force_constant13
    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 2*2 * 3])  # 解的矩阵

    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值 (0,1700) (0,rt3)
            # print('1 >> ', i)
            kx = i * 2 * np.pi / 3 / a / (n * int(10 * 3 ** 0.5))
            ky = 0
            # plt.scatter(kx, ky)
        elif i < (10 + int(10 * 3 ** 0.5)) * n:
            # print('2 >> ', i)  # (1700,2700) (rt3,rt3+1)
            kx = 2 * np.pi / 3 / a
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a / 3 ** 0.5
            # plt.scatter(kx, ky)
        else:
            # print('3 >> ', i)  # (2700,4700) (rt3+1,rt3+3)
            kx = 2 * np.pi / 3 / a - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a
            ky = kx / 3 ** 0.5
            # plt.scatter(kx, ky)
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
    plt.plot(kk, result, lw=2)
    plt.xticks(xk, ["Γ", "K", "M", "Γ"])
    # plt.xlim(0, s + 1)
    # plt.ylim(0, 15)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(s, color='gray', linestyle='--')
    plt.axvline(s + 1, color='gray', linestyle='--')
    plt.title('%.2f ° ' % angle)
    # plt.savefig('png/声子色散.png', dpi=200)
    plt.show()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
