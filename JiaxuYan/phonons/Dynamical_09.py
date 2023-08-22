#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Dynamical_09.py    
@Time    :   2023/3/6 16:52  
@E-mail  :   iamwxyoung@qq.com
@Tips    :
    这是后来的那个2X2的文献
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
    alpha1 = 445.0  # N/m
    alpha2 = 102.0  # N/m
    # 来自文献的几个公式，用于构建D矩阵

    dA0 = 3/2*alpha1 + 3*alpha2*(1-np.cos(v_[0]*a_*rt3/2)*np.cos(v_[1]*a_/2))
    dB0 = -alpha2*rt3*np.sin(v_[0]*a_*rt3/2)*np.sin(v_[1]*a_/2)
    dC0 = -alpha1*(np.exp(-1j*(v_[0]*a_/rt3)) + 1/2*np.exp(1j*(v_[0]*a_/2/rt3))*np.cos(v_[1]*a_/2))
    dD0 = -1j*rt3/2*alpha1*np.exp(1j*(v_[0]*a_/2/rt3))*np.sin(v_[1]*a_/2)
    dA1 = 3/2*alpha1+alpha2*(3-2*np.cos(v_[1]*a_)-np.cos(v_[0]*a_*rt3/2)*np.cos(v_[1]*a_/2))
    dB1 = -3/2*alpha1*np.exp(1j*(v_[0]*a_/2/rt3))*np.cos(v_[1]*a_/2)
    dC1 = np.conj(dC0).T
    dD1 = np.conj(dD0).T
    dB2 = np.conj(dB1).T

    Dlist = [dA0, dB0, dC0, dD0,
             dB0, dA1, dD0, dB1,
             dC1, dD1, dA0, dB0,
             dD1, dB2, dB0, dA1]
    Dm = np.array(Dlist).reshape((4, 4))
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
    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 2 * 2])  # 解的矩阵
    # Kgamma = (0, 0) Km = (2*π/a/rt3, 0) Kk = (2π/a/rt3, 2π/a/3)
    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值 (0,1700) (0,rt3) print('1 >> ', i)
            kx = i * 2 * np.pi / 3 / a_cc / (n * int(10 * 3 ** 0.5))
            ky = 0
        elif i < (10 + int(10 * 3 ** 0.5)) * n:  # print('2 >> ', i)  # (1700,2700) (rt3,rt3+1)
            kx = 2 * np.pi / 3 / a_cc
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a_cc / 3 ** 0.5
        else:  # print('3 >> ', i)  # (2700,4700) (rt3+1,rt3+3)
            kx = 2 * np.pi / 3 / a_cc - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a_cc
            ky = kx / 3 ** 0.5
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
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    # plt.xlim(0, s + 1)
    # plt.ylim(0, 15)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(xk[1], color='gray', linestyle='--')
    plt.axvline(xk[2], color='gray', linestyle='--')
    plt.title('09')
    plt.show()
    print('finished')
