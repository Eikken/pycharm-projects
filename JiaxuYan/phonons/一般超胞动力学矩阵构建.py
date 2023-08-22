#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   一般超胞动力学矩阵构建.py    
@Time    :   2023/2/15 16:15  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   我们认为此超胞扩胞方法行不通×
基本二维变换有比例变换（Scaling）、旋转变换（Rotating）、错切变换（Shearing）和平移变换（Translating）。
    # / |  A  |  B  |  C  |  D  |
    # --|-----|-----|-----|-----|
    # A | A A | A B | A C | A D |
    # --|-----|-----|-----|-----|
    # B | B A | B B | B C | B D |
    # --|-----|-----|-----|-----|
    # C | C A | C B | C C | C D |
    # --|-----|-----|-----|-----|
    # D | D A | D B | D C | D D |
    # --|-----|-----|-----|-----|
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def tmx(theta_):  # twist matrix
    return np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])


def rotateFunc(theta_, r_):
    # 现在开始扩展为4 atoms 的model
    # r_ = np.array([[1], [0], [0]])
    # x+=1/2, y+=rt3/2
    tr_ = np.c_[r_, 1].reshape(1, np.size(r_, 0)+1)
    # r_ = [1, 0, 0, 1]
    atom_nexus_0 = []  # 第0个原子不需要变换
    atom_nexus_1 = []  # 第1个原子的变换关系：tmx(theta + pi)
    atom_nexus_2 = []  # 第2个原子的变换关系，齐次坐标变换平移Tr
    atom_nexus_3 = []  # 第3个原子的变换关系，tmx(theta + pi) -> atom_2
    r0_, r1_, r2_, r3_ = 0, 0, 0, 0
    Um = tmx(theta_)
    Upi = tmx(np.pi)
    tx_, ty_ = 3 / 2, rt3 / 2
    Trans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [tx_, ty_, 0, 1]])
    for i in range(int(2 * np.pi / theta_)):
        if np.size(r_) == 3:
            r0_ = np.matmul(np.linalg.inv(Um), r_)
            r1_ = r0_ * -1
            r2_ = np.matmul(np.linalg.inv(Um), np.matmul(tr_, Trans).reshape(np.size(tr_, 1), np.size(tr_, 0))[:3])
            r3_ = r2_ * -1
        else:
            r0_ = np.linalg.inv(Um) @ r_ @ Um  # @ 表示矩阵乘法(不常用)
            r1_ = np.linalg.inv(Upi) @ r0_ @ Upi
            r2_ = np.linalg.inv(Um) @ np.matmul(r_, Trans) @ Um  # @ 表示矩阵乘法(不常用)
            r3_ = np.linalg.inv(Upi) @ r2_ @ Upi
        atom_nexus_0.append(r0_)
        atom_nexus_1.append(r1_)
        atom_nexus_2.append(r2_)
        atom_nexus_3.append(r3_)
    return [atom_nexus_0, atom_nexus_1, atom_nexus_2, atom_nexus_3]


def K_matrix(theta_, k_):
    U = np.array([[np.cos(theta_), np.sin(theta_), 0], [-np.sin(theta_), np.cos(theta_), 0], [0, 0, 1]])
    return np.linalg.inv(U) @ k_ @ U  # K矩阵左乘右乘rotate矩阵


def D_matrix(k_):
    D = np.zeros([4 * 3, 4 * 3], dtype=np.complex)

    # D00, D01, D02, D03 = 0, 0, 0, 0
    # D10, D11, D12, D13 = 0, 0, 0, 0
    # D20, D21, D22, D23 = 0, 0, 0, 0
    # D30, D31, D32, D33 = 0, 0, 0, 0
    DAAs, DBBs, DBAs, DABs = 0, 0, 0, 0

    for i in range(3):
        DAAs = DAAs + \
               Sphere_2[0][i] * np.exp(-1j * np.matmul(k_, Second_0123[0][i])) + \
               Sphere_2[0][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[0][i + 3])) + \
               Sphere_2[1][i] * np.exp(-1j * np.matmul(k_, Second_0123[1][i])) + \
               Sphere_2[1][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[1][i + 3]))
        DBBs = DBBs + \
               Sphere_2[2][i] * np.exp(-1j * np.matmul(k_, Second_0123[2][i])) + \
               Sphere_2[2][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[2][i + 3])) + \
               Sphere_2[3][i] * np.exp(-1j * np.matmul(k_, Second_0123[3][i])) + \
               Sphere_2[3][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[3][i + 3]))

        DABs = DABs + \
               Sphere_1[0][i] * np.exp(-1j * np.matmul(k_, First_0123[0][i])) + \
               Sphere_1[1][i] * np.exp(-1j * np.matmul(k_, First_0123[1][i])) + \
               Sphere_3[0][i] * np.exp(-1j * np.matmul(k_, Third_0123[0][i])) + \
               Sphere_3[1][i] * np.exp(-1j * np.matmul(k_, Third_0123[1][i])) + \
               Sphere_4[0][i] * np.exp(-1j * np.matmul(k_, Four_0123[0][i])) + \
               Sphere_4[1][i] * np.exp(-1j * np.matmul(k_, Four_0123[1][i])) + \
               Sphere_4[0][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[0][i + 3])) + \
               Sphere_4[1][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[1][i + 3]))

        DBAs = DBAs + \
               Sphere_1[2][i] * np.exp(-1j * np.matmul(k_, First_0123[2][i])) + \
               Sphere_1[3][i] * np.exp(-1j * np.matmul(k_, First_0123[3][i])) + \
               Sphere_3[2][i] * np.exp(-1j * np.matmul(k_, Third_0123[2][i])) + \
               Sphere_3[3][i] * np.exp(-1j * np.matmul(k_, Third_0123[3][i])) + \
               Sphere_4[2][i] * np.exp(-1j * np.matmul(k_, Four_0123[2][i])) + \
               Sphere_4[3][i] * np.exp(-1j * np.matmul(k_, Four_0123[3][i])) + \
               Sphere_4[2][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[2][i + 3])) + \
               Sphere_4[3][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[3][i + 3]))
        # D00 = D00 + Sphere_2[0][i] * np.exp(-1j * np.matmul(k_, Second_0123[0][i])) + \
        #       Sphere_2[0][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[0][i + 3]))
        # D11 = D11 + Sphere_2[1][i] * np.exp(-1j * np.matmul(k_, Second_0123[1][i])) + \
        #       Sphere_2[1][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[1][i + 3]))
        # D22 = D22 + Sphere_2[2][i] * np.exp(-1j * np.matmul(k_, Second_0123[2][i])) + \
        #       Sphere_2[2][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[2][i + 3]))
        # D33 = D33 + Sphere_2[3][i] * np.exp(-1j * np.matmul(k_, Second_0123[3][i])) + \
        #       Sphere_2[3][i + 3] * np.exp(-1j * np.matmul(k_, Second_0123[3][i + 3]))
        # D01 = D01 + \
        #       Sphere_1[1][i] * np.exp(-1j * np.matmul(k_, First_0123[1][i])) + \
        #       Sphere_3[1][i] * np.exp(-1j * np.matmul(k_, Third_0123[1][i])) + \
        #       Sphere_4[1][i] * np.exp(-1j * np.matmul(k_, Four_0123[1][i])) + \
        #       Sphere_4[1][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[1][i + 3]))
        # D02 = D02 + \
        #       Sphere_1[2][i] * np.exp(-1j * np.matmul(k_, First_0123[2][i])) + \
        #       Sphere_3[2][i] * np.exp(-1j * np.matmul(k_, Third_0123[2][i])) + \
        #       Sphere_4[2][i] * np.exp(-1j * np.matmul(k_, Four_0123[2][i])) + \
        #       Sphere_4[2][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[2][i + 3]))
        # D03 = D03 + \
        #       Sphere_1[3][i] * np.exp(-1j * np.matmul(k_, First_0123[3][i])) + \
        #       Sphere_3[3][i] * np.exp(-1j * np.matmul(k_, Third_0123[3][i])) + \
        #       Sphere_4[3][i] * np.exp(-1j * np.matmul(k_, Four_0123[3][i])) + \
        #       Sphere_4[3][i + 3] * np.exp(-1j * np.matmul(k_, Four_0123[3][i + 3]))

    D[0:3, 3:6] = -DABs
    D[3:6, 0:3] = -DBAs
    D[0:3, 0:3] = sum([sum(i) + sum(j) for i, j in zip(Sphere_1[0], Sphere_1[1])]) + \
                  sum([sum(i) + sum(j) for i, j in zip(Sphere_3[0], Sphere_3[1])]) + \
                  sum([sum(i) + sum(j) for i, j in zip(Sphere_4[0], Sphere_4[1])]) - DAAs
    D[3:6, 3:6] = sum([sum(i) + sum(j) for i, j in zip(Sphere_1[2], Sphere_1[3])]) + \
                  sum([sum(i) + sum(j) for i, j in zip(Sphere_3[2], Sphere_3[3])]) + \
                  sum([sum(i) + sum(j) for i, j in zip(Sphere_4[2], Sphere_4[3])]) - DBBs
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
    a = 0.142  # 1.42e-10
    m = 1.99  # 1.99e-26
    rt3 = 3 ** 0.5
    f = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])
    # A为圆心周围的原子坐标及B的坐标
    First_0123 = rotateFunc(2 / 3 * np.pi, np.array([[1], [0], [0]]) * a)
    Second_0123 = rotateFunc(1 / 3 * np.pi, np.array([[3 / 2], [3 ** 0.5 / 2], [0]]) * a)
    Third_0123 = rotateFunc(2 / 3 * np.pi, np.array([[1], [3 ** 0.5], [0]]) * a)
    Four_0123_up = rotateFunc(2 / 3 * np.pi, np.array([[2.5], [3 ** 0.5 / 2], [0]]) * a)
    Four_0123_down = rotateFunc(2 / 3 * np.pi, np.array([[2.5], [-3 ** 0.5 / 2], [0]]) * a)
    Four_0123 = [i + j for i, j in zip(Four_0123_up, First_0123)]
    # return list value like : [A_0, B_1, A_2, B_3]

    # 得到初始K矩阵，角度由文献的图9.1 可知，分别得到（a）图和(b)图情况的初始K矩阵
    Sphere_1 = rotateFunc(2 / 3 * np.pi, f[0:3, 0:3])  # 这里应该叫 # AB1、BA1、CD1、DC1
    Sphere_2 = rotateFunc(1 / 3 * np.pi, K_matrix(1 / 6 * np.pi, f[3:6, 3:6]))  # AC2 \ CA2 \ BD2 \ DB2 A=C\B=D
    # question k矩阵的π/6 、 π/3 等是怎么确定给的？是相对于初始AB1的角度，
    # 也就是先转自己的K矩阵到初始位置，然后再把K矩阵旋转得到这层周围所有的点
    Sphere_3 = rotateFunc(2 / 3 * np.pi, K_matrix(1 / 3 * np.pi, f[6:9, 6:9]))  # AD3 \ DA3 \ BC3 \ CB3
    Sphere_4_up = rotateFunc(2 / 3 * np.pi, K_matrix(np.arccos(2.5 / 7 ** 0.5), f[9:12, 9:12]))
    # AD4 \ DA4 \ BC4 \ CB4
    Sphere_4_down = rotateFunc(2 / 3 * np.pi, K_matrix(2 * np.pi - np.arccos(2.5 / 7 ** 0.5), f[9:12, 9:12]))
    Sphere_4 = [i + j for i, j in zip(Sphere_4_up, Sphere_4_down)]
    # AB1 \ BA1 \ CD1 \ DC1
    # AC2 \ CA2 \ BD2 \ DB2     A = C \ B = D
    # AD3 \ DA3 \ BC3 \ CB3
    # AD4 \ DA4 \ BC4 \ CB4
    n = 100
    res = cal_matrix()
    # 沿第一布里渊区的最小重复单元的边界每条边单位长度取100个k点进行计算（主要是为了使点均匀分布，同样可以每条边取n 个点进行计算）
    xk = [0, rt3, rt3 + 1, rt3 + 3]
    kk = np.linspace(0, 4.7, num=(30 + int(rt3 * 10)) * n)  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(4, 5))
    plt.plot(kk, res, c="r")
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    plt.xlim(0, 4.75)
    plt.ylim(0, 1e14)
    plt.ylabel("ω", fontsize=14)
    plt.axvline(rt3, color='gray', linestyle='--')
    plt.axvline(rt3 + 1, color='gray', linestyle='--')

    plt.show()
