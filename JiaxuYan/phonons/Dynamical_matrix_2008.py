#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Dynamical_matrix_2008.py    
@Time    :   2023/2/27 16:28  
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
    fcs = np.array(cf.force_constant_98)

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

    # # # f5
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
    # # phi00 =
    # fijAA = f1aa + f2aa + f3aa + f4aa + f5aa
    # fijAB = f1ab + f2ab + f3ab + f4ab + f5ab

    fijAA = f1aa + f2aa + f3aa + f4aa
    fijAB = f1ab + f2ab + f3ab + f4ab

    fijBB = fijAA
    fijBA = np.conj(-fijAB).T

    Dm = np.zeros((2 * 3, 2 * 3), dtype=complex)
    Dm[0:3, 0:3] = fijAA
    Dm[3:6, 3:6] = fijBB
    Dm[0:3, 3:6] = -fijAB
    Dm[3:6, 0:3] = fijBA
    return Dm


if __name__ == '__main__':
    # start here
    rt3 = 3 ** 0.5

    a_cc = 1.42
    a = a_cc * rt3
    M = 12.02 * 1.66 * 1e-24  # g
    m = 12

    n = 200
    angle = 0

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 2 * 3])  # 解的矩阵

    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值 (0,1700) (0,rt3)
            # print('1 >> ', i)
            kx = i * 2 * np.pi / 3 / a_cc / (n * int(10 * 3 ** 0.5))
            ky = 0
            # plt.scatter(kx, ky)
        elif i < (10 + int(10 * 3 ** 0.5)) * n:
            # print('2 >> ', i)  # (1700,2700) (rt3,rt3+1)
            kx = 2 * np.pi / 3 / a_cc
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a_cc / 3 ** 0.5
            # plt.scatter(kx, ky)
        else:
            # print('3 >> ', i)  # (2700,4700) (rt3+1,rt3+3)
            kx = 2 * np.pi / 3 / a_cc - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a_cc
            ky = kx / 3 ** 0.5
            # plt.scatter(kx, ky)
        k = np.array([kx, ky])  # 得到k值，带入D矩阵
        dm = dym08(theta=np.deg2rad(angle), v=k)
        w, t = np.linalg.eig(dm)
        w = list(w)
        w.sort()
        result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存
    # for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
    #     if i < 2000:  # 判断i的大小确定k的取值
    #         # (0,2000) (0,2)
    #         kx = i / (n * 20 - 1) * 2 * np.pi / 3 / a_cc
    #         ky = kx / 3 ** 0.5
    #
    #     elif i < 3000:
    #         # print('2 >> ', i)  # (2000,3000) (2,3)
    #         kx = 2 * np.pi / 3 / a_cc
    #         ky = (n * int(10 * 3) - i) / (10 * n - 1) * 2 * np.pi / 3 / a_cc / 3 ** 0.5
    #     else:
    #         # print('3 >> ', i)  # (3000,4700) (3,3+rt3)
    #         kx = (n * int(30 + 10 * rt3) - i) / (n * int(10)) * 2 * np.pi / 3 / a_cc / rt3
    #         ky = 0
    #     k = np.array([kx, ky])# 得到k值，带入D矩阵
    #
    #     dm = dym08(v=k)
    #     w, t = np.linalg.eig(dm)
    #     w = list(w)
    #     w.sort()
    #     result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存
    # plt.show()
    # xk = [0, 2, 3, rt3 + 3]
    xk = [0, rt3, rt3 + 1, rt3 + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(3 ** 0.5 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(kk, result, lw=2)
    plt.xticks(xk, ["Γ", "K", "M", "Γ"])
    # plt.xlim(0, s + 1)
    # plt.ylim(0, 15)
    # plt.ylabel("ω", fontsize=14)
    plt.axvline(xk[1], color='gray', linestyle='--')
    plt.axvline(xk[2], color='gray', linestyle='--')
    # plt.title('%.2f ° ' % angle)
    plt.tick_params(labelsize=18)
    # plt.savefig('png/声子色散.png', dpi=200)
    plt.show()
    print('finished')
