#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   动力学计算声子谱.py
@Time    :   2023/2/10 9:26  
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


def rotateFunc(theta_, r_):
    R = []
    RB = []
    Um = np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])
    U = np.array([[np.cos(np.pi), np.sin(np.pi), 0], [-np.sin(np.pi), np.cos(np.pi), 0], [0, 0, 1]])
    for i in range(int(2 * np.pi / theta_)):
        if np.size(r_) == 3:
            r_ = np.matmul(np.linalg.inv(Um), r_)
            rb_ = r_ * -1
        else:
            r_ = np.linalg.inv(Um) @ r_ @ Um  # @ 表示矩阵乘法(不常用)
            rb_ = np.linalg.inv(U) @ r_ @ U
        R.append(r_)
        RB.append(rb_)
    return R, RB


def K_matrix(theta_, k_):
    U = np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])
    return np.linalg.inv(U) @ k_ @ U  # K矩阵左乘右乘rotate矩阵


def D_matrix(k_):
    D = np.zeros([2*3, 2*3], dtype=np.complex)
    DAAs, DBBs, DBAs, DABs = 0, 0, 0, 0
    for i in range(3):
        # 这里的 exp(-i*Kij)中，负号在1j或者k_或者SecA都可以。
        DAAs = DAAs + KAA[i] * np.exp(-1j * np.matmul(k_, SecA[i])) + KAA[i + 3] * np.exp(
            -1j * np.matmul(k_, SecA[i + 3]))
        DBBs = DBBs + KBB[i] * np.exp(-1j * np.matmul(k_, SecB[i])) + KBB[i + 3] * np.exp(
            -1j * np.matmul(k_, SecB[i + 3]))

        DABs = DABs + KAB1[i] * np.exp(1j * np.matmul(k_, -FirA[i])) + \
               KAB3[i] * np.exp(1j * np.matmul(k_, -ThiA[i])) + \
               KAB4[i] * np.exp(1j * np.matmul(k_, -FouA[i])) + \
               KAB4[i + 3] * np.exp(1j * np.matmul(k_, -FouA[i + 3]))
        DBAs = DBAs + KBA1[i] * np.exp(1j * np.matmul(k_, -FirB[i])) + \
               KBA3[i] * np.exp(1j * np.matmul(k_, -ThiB[i])) + \
               KBA4[i] * np.exp(1j * np.matmul(k_, -FouB[i])) + \
               KBA4[i + 3] * np.exp(1j * np.matmul(k_, -FouB[i + 3]))
    D[0:3, 3:6] = -DABs
    D[3:6, 0:3] = -DBAs
    D[0:3, 0:3] = sum([sum(KAB1), sum(KAA),  sum(KAB3),  sum(KAB4f),  sum(KAB4s)]) - DAAs  # sum为同纬度矩阵中对应元素相加
    D[3:6, 3:6] = sum([sum(KAB1),  sum(KAA), sum(KAB3),  sum(KAB4f),  sum(KAB4s)]) - DBBs
    # sum(a) + sum(b) + sum(c) == sum([sum(a), sum(b), sum(c)])

    return D


def cal_matrix():
    # 求解矩阵

    result = np.zeros([(30 + int(3 ** 0.5 * 10)) * n, 6])  # 解的矩阵
    for i in range((30 + int(3 ** 0.5 * 10)) * n):  # 在这里将sqrt(3)近似取为17，没有什么特别的意义
        if i < n * int(10 * 3 ** 0.5):  # 判断i的大小确定k的取值
            kx = i * 2 * np.pi / 3 / a / (n * int(10 * 3 ** 0.5))
            ky = 0
        elif i < (10 + int(10 * 3 ** 0.5)) * n:
            kx = 2 * np.pi / 3 / a
            ky = (i - n * int(10 * 3 ** 0.5)) / (10 * n - 1) * 2 * np.pi / 3 / a / 3 ** 0.5
        else:
            kx = 2 * np.pi / 3 / a - (i - (10 + int(10 * 3 ** 0.5)) * n) / (n * 20 - 1) * 2 * np.pi / 3 / a
            ky = kx / 3 ** 0.5
        k = np.array([kx, ky, 0])  # 得到k值，带入D矩阵
        w, t = np.linalg.eig(D_matrix(k))
        w = list(w)
        w.sort()
        result[i, :] = (np.real(np.sqrt(w) / m ** 0.5))  # 将本征值进行保存

    return result


if __name__ == '__main__':
    # start here
    a = 1.42e-10
    m = 1.99e-26
    f = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])
    # A为圆心周围的原子坐标及B的坐标
    FirA, FirB = rotateFunc(2 / 3 * np.pi, np.array([[1], [0], [0]]) * a)
    SecA, SecB = rotateFunc(1 / 3 * np.pi, np.array([[3 / 2], [3 ** 0.5 / 2], [0]]) * a)
    ThiA, ThiB = rotateFunc(2 / 3 * np.pi, np.array([[1], [3 ** 0.5], [0]]) * a)
    FouA1, FouB1 = rotateFunc(2 / 3 * np.pi, np.array([[2.5], [3 ** 0.5 / 2], [0]]) * a)
    FouA2, FouB2 = rotateFunc(2 / 3 * np.pi, np.array([[2.5], [-3 ** 0.5 / 2], [0]]) * a)
    FouA = FouA1 + FouA2
    FouB = FouB1 + FouB2

    # 得到初始K矩阵，角度由文献的图9.1 可知，分别得到（a）图和(b)图情况的初始K矩阵
    KAB1, KBA1 = rotateFunc(2 / 3 * np.pi, f[0:3, 0:3])
    KAA, KBB = rotateFunc(1 / 3 * np.pi, K_matrix(1 / 6 * np.pi, f[3:6, 3:6]))
    # KAAK = K_matrix(1 / 6 * np.pi, f[3:6, 3:6])
    # question k矩阵的π/6 、 π/3 等是怎么确定给的？是相对于初始AB1的角度，
    # 第一层和第二层A - B1 - Fir与A - A1 - Sec是π / 6 ；第一层和第三层A - B1 - Fir与A - B1 - Thi是π / 3
    # 也就是先转自己的K矩阵到初始位置，然后再把K矩阵旋转得到这层周围所有的点
    KAB3, KBA3 = rotateFunc(2 / 3 * np.pi, K_matrix(1 / 3 * np.pi, f[6:9, 6:9]))
    KAB4f, KBA4f = rotateFunc(2 / 3 * np.pi, K_matrix(np.arccos(2.5 / 7 ** 0.5), f[9:12, 9:12]))
    KAB4s, KBA4s = rotateFunc(2 / 3 * np.pi, K_matrix(2 * np.pi - np.arccos(2.5 / 7 ** 0.5), f[9:12, 9:12]))
    KAB4 = KAB4f + KAB4s
    KBA4 = KBA4f + KBA4s

    s = 3 ** 0.5
    n = 100
    res = cal_matrix()
    # 沿第一布里渊区的最小重复单元的边界每条边单位长度取100个k点进行计算（主要是为了使点均匀分布，同样可以每条边取n 个点进行计算）
    xk = [0, s, s + 1, s + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(3 ** 0.5 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(4, 5))
    plt.plot(kk, res, c="r")
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    plt.xlim(0, 4.75)
    plt.ylim(0, 1e14)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(s, color='gray', linestyle='--')
    plt.axvline(s + 1, color='gray', linestyle='--')

    plt.show()
