#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Appendix_B_2008.py    
@Time    :   2023/3/2 15:12  
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
import JiaxuYan.phonons.constant_file as cf


def dym08(*args, **kwargs):

    fcs = np.array(cf.force_constant_my)

    a_ = a
    v_ = np.c_[kwargs['v'].reshape(1, 2), 0].reshape(1, 3)[0, :]  # [qx, qy, qz]
    # v_ = [2 * np.pi / 3 / a_cc, 2 * np.pi / 3 / rt3 / a_cc]
    f1xxaa = 3 * (fcs[0][0] + fcs[0][1]) / 2  # 1 代表n=1圈
    f1zzaa = 3 * fcs[0][2]
    f1aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f1aa[0, 0] = f1xxaa
    f1aa[1, 1] = f1xxaa
    f1aa[2, 2] = f1zzaa

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
    # f1ab[1, 0] = f1xyab  # np.conj(f1xyab).T
    # f2
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
    # f2aa[1, 0] = f2xyaa  # np.conj(f2xyaa).T
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
    # f3ab[1, 0] = f3xyab  # np.conj(f3xyab).T

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
    # f4ab[1, 0] = f4xyab  # np.conj(f4xyab).T

    # # f5
    f5xxaa = 1 * ((fcs[4][0] + 3 * fcs[4][1]) * (
            np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
            rt3 * a_) / 4) + 4 *
                  fcs[4][
                      0] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    f5yyaa = 1 * ((fcs[4][1] + 3 * fcs[4][0]) * (
            np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
            rt3 * a_) / 4) + 4 *
                  fcs[4][
                      0] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    f5zzaa = 1 * ((fcs[4][2] + 3 * fcs[4][2]) * (
            np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 + np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (
            rt3 * a_) / 4) + 4 *
                  fcs[4][
                      2] * np.sin(v_[0] * (rt3 * a_) / 2) ** 2)
    f5xyaa = -rt3 * (fcs[4][0] - fcs[4][1]) * (
            np.sin(rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4 - np.sin(-rt3 * v_[1] + v_[0]) ** 2 * (rt3 * a_) / 4)
    f5aa = np.zeros((3, 3), dtype=complex)  # 其他都是0
    f5aa[0, 0] = f5xxaa
    f5aa[1, 1] = f5yyaa
    f5aa[2, 2] = f5zzaa
    f5aa[0, 1] = f5xyaa
    # f5aa[1, 0] = f5xyaa  # np.conj(f5xyaa).T
    f5ab = f2ab  # 其他都是0

    fijAA = f1aa + f2aa + f3aa + f4aa + f5aa
    fijAB = f1ab + f2ab + f3ab + f4ab + f5ab

    # fijAA = f1aa + f2aa + f3aa + f4aa
    # fijAB = f1ab + f2ab + f3ab + f4ab

    fijBB = fijAA
    fijBA = np.conj(fijAB).T

    # Ds is single layer A-B site matrix
    # # hii >> ii[0]    22[1]    33[2]   12[3]    13[4]     23[5]
    h1sn = [-114.47, -114.47, 2040.41, 0.00, 0.00, 0.00]  # 1+, sn = single none
    h1sp = [63.1700, -17.41, 430.87, 0.00, -190.05, 0.00]  # 1+', sp = single pie
    h2sp = [2.73000, 43.02, 430.87, -34.89, 95.03, -164.59]  # 2+', sp = single pie
    h1dp = [8.95000, 7.19, 12.84, 1.52, 4.15, 2.40]  # 1+", dp = double pie
    h2dp = [6.31000, 9.83, 12.84, 0, 0, 4.79]  # 2+", dp = double pie

    c_ = 6.7
    w_ = np.array([v_[0], v_[1], c_ / 2])  # 没啥，占用一下内存，多一个引用而已。
    # hij is force constants given by empirical Lennard-Jones potential
    # The form above is hij form, the data come from paper 2008.
    #  # hxxaasp = 0  ## h is H; xx is ii (i in range(3)); aap is AA', p is single ' .
    hii = []
    for j in range(3):
        val = -2 * (h1sn[j] + 2 * (2 * h1dp[j] * np.cos(w_[0] * rt3 * a_ / 2) * np.cos(w_[1] * a_ / 2) +
                                   h2dp[j] * np.cos(w_[1] * a_))) * np.cos(w_[2] * c_ / 2)
        hii.append(val)
    [hxxaap, hyyaap, hzzaap] = [j for j in hii]
    hxyaap = 8 * h1dp[3] * np.sin(w_[0] * rt3 * a_ / 2) * np.sin(w_[1] * a_ / 2) * np.cos(w_[2] * c_ / 2)
    hxzaap = 8 * h1dp[4] * np.sin(w_[0] * rt3 * a_ / 2) * np.cos(w_[1] * a_ / 2) * np.sin(w_[2] * c_ / 2)
    hyzaap = 4 * (2 * (h1dp[5] * np.sin(w_[1] * a_ / 2) * np.cos(w_[0] * rt3 * a_ / 2)) + h2dp[5] * np.sin(
        w_[1] * a_)) * np.sin(w_[2] * c_ / 2)

    haap = np.zeros((3, 3), dtype=complex)
    haap[0, 0] = hxxaap
    haap[1, 1] = hyyaap
    haap[2, 2] = hzzaap
    haap[0, 1] = hxyaap
    haap[0, 2] = hxzaap
    haap[1, 2] = hyzaap

    hii = []
    for j in range(3):
        val = -2 * (h1sp[j] * np.exp(-1j * w_[0] * a_ / rt3) + 2 * h2sp[j] * np.cos(w_[1] * a_ / 2) * np.exp(
            1j * w_[0] * a_ / 2 / rt3)) * np.cos(w_[2] * c_ / 2)
        hii.append(val)
    [hxxabp, hyyabp, hzzabp] = [j for j in hii]
    hxyabp = -4j * h2sp[3] * np.sin(w_[1] * a_ / 2) * np.exp(1j * w_[0] * a_ / 2 / rt3) * np.cos(w_[2] * c_ / 2)
    hxzabp = -2j * (h1sp[4] * np.exp(-1j * w_[0] * a_ / rt3) + 2 * h2sp[4] * np.exp(1j * w_[0] * a_ / 2 / rt3) * np.cos(
        w_[1] * a_ / 2)) * np.sin(w_[2] * c_ / 2)
    hyzabp = 4 * h2sp[5] * np.exp(1j * w_[0] * a_ / 2 / rt3) * np.sin(w_[1] * a_ / 2) * np.sin(w_[2] * c_ / 2)

    habp = np.zeros((3, 3), dtype=complex)
    habp[0, 0] = hxxabp
    habp[1, 1] = hyyabp
    habp[2, 2] = hzzabp
    habp[0, 1] = hxyabp
    habp[0, 2] = hxzabp
    habp[1, 2] = hyzabp

    hbbp = np.conj(habp).T  # B5
    hbap = habp  # B6

    hAA = -1 / 2 * (haap + habp)
    hBB = -1 / 2 * (hbap + hbbp)

    hapa = haap
    hbpb = np.conj(hbbp).T  # 注 共轭矩阵是相互的
    hbap = habp  # np.conj(habp).T

    #  # Dym alpha alpha
    fAA = fijAA + 2 * hAA
    fBB = fijBB + 2 * hBB
    fAB = fijAB
    fBA = fijBA

    #  # Dym alpha beta
    hAA = haap
    hBB = hbbp
    hAB = habp
    hBA = hbap

    #  # Dym alpha alpha

    fAA = fijAA + 2 * hAA
    fBB = fijBB + 2 * hBB
    fAB = fijAB
    fBA = fijBA

    Ddouble = np.zeros((2 * 2 * 3, 2 * 2 * 3), dtype=complex)
    Daa = np.zeros((2 * 3, 2 * 3), dtype=complex)
    Daa[0:3, 0:3] = fAA
    Daa[3:6, 3:6] = fBB
    Daa[0:3, 3:6] = fAB
    Daa[3:6, 0:3] = fBA

    Dab = np.zeros((2 * 3, 2 * 3), dtype=complex)
    Dab[0:3, 0:3] = hAA
    Dab[3:6, 3:6] = hBB
    Dab[0:3, 3:6] = hAB
    Dab[3:6, 0:3] = hBA

    # Ddouble = np.zeros((2 * 2 * 3, 2 * 2 * 3), dtype=complex)
    Dbb = np.zeros((2 * 3, 2 * 3), dtype=complex)
    Dbb[0:3, 0:3] = fAA
    Dbb[3:6, 3:6] = fBB
    Dbb[0:3, 3:6] = fAB
    Dbb[3:6, 0:3] = fBA

    Dbb = np.conj(Daa).T
    Dba = np.conj(Dab).T

    Ddouble[0:6, 0:6] = Daa
    Ddouble[6:12, 6:12] = Dbb
    Ddouble[0:6, 6:12] = Dab
    Ddouble[6:12, 0:6] = Dba

    # note
    return Ddouble


if __name__ == '__main__':
    # start here
    # start here
    rt3 = 3 ** 0.5

    a_cc = 1.42
    a = a_cc * rt3
    M = 12.02 * 1.66 * 1e-24  # g
    m = 12

    n = 200
    angle = 0

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 2 * 2 * 3])  # 解的矩阵

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
        dm = dym08(theta=np.deg2rad(angle), v=k)
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
