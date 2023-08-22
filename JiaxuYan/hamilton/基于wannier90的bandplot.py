#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于wannier90的bandplot.py    
@Time    :   2023/3/31 15:12  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os

num_wan = 5
basis_vector = [[1.37287871, 1.37287871, -2.74575742], [-2.74575742, 1.37287871, 1.37287871],
                [13.36629497, 13.36629497, 13.36629497]]
E_fermi = -1.3286
K_point_path = [[0, 0, 0], [0.50000, 0.00000, 0.00000], [0.33333, 0.33333, 0.00000], [0.00000, 0.00000, 0.00000]]
k_meshes = [40, 30, 40]

Symmetry_point_label1 = "G"
Symmetry_point_label2 = "M"
Symmetry_point_label3 = "K"
Symmetry_point_label4 = "G"

V = np.dot(basis_vector[0], np.cross(basis_vector[1], basis_vector[2]))
rec = [np.cross(basis_vector[1], basis_vector[2]) * 2 * np.pi / V,
       np.cross(basis_vector[2], basis_vector[0]) * 2 * np.pi / V,
       np.cross(basis_vector[0], basis_vector[1]) * 2 * np.pi / V]

for i in range(len(K_point_path)):
    K_point_path[i] = K_point_path[i][0] * rec[0] + K_point_path[i][1] * rec[1] + K_point_path[i][2] * rec[2]


def k_path():
    k_point = []
    for i in range(len(K_point_path) - 1):
        interval = np.array(K_point_path[i + 1]) - np.array(K_point_path[i])
        interval = interval / k_meshes[i]
        for j in range(k_meshes[i]):
            k_point.append(np.array(K_point_path[i]) + j * interval)
    return k_point


def phase(R1, R2, R3, k1, k2, k3):
    R1_vector = R1 * np.array(basis_vector[0])
    R2_vector = R2 * np.array(basis_vector[1])
    R3_vector = R3 * np.array(basis_vector[2])
    R_vec = R1_vector + R2_vector + R3_vector
    inner_product = np.dot(R_vec, [k1, k2, k3])
    return np.exp(1j * inner_product)


def read_Hr_dat():
    """
    T.Shape = num_wan, num_wan, n, 1
    R.Shape = num_wan, num_wan, n, 3
    # 是hopping总数
    """
    with open("data/wannier90_hr_SVO.dat", "r") as f:
        lines = f.readlines()
    R = [[[] for col in range(num_wan)] for row in range(num_wan)]
    T = [[[] for col in range(num_wan)] for row in range(num_wan)]
    for line in lines:
        if len(line.split()) == 7:
            r1, r2, r3, n1_f, n2_f, t_real, t_image = list(map(float, line.split()))
            n1, n2 = int(n1_f) - 1, int(n2_f) - 1
            R[n1][n2].append([r1, r2, r3])
            T[n1][n2].append([t_real + 1j * t_image])
    return T, R


def matrix_construct(factor, R, k1, k2, k3):
    H = np.zeros((num_wan, num_wan), dtype='complex')
    for i in range(num_wan):
        for j in range(num_wan):
            for k in range(len(R[i][j])):  # Hij 是ij处所有的相位因子和Rij矢量的乘积求和。
                H[i][j] = H[i][j] + factor[i][j][k][0] * phase(R[i][j][k][0], R[i][j][k][1], R[i][j][k][2], k1, k2, k3)
    return H


if __name__ == '__main__':
    norm_qb = 2.46
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
    k_line = k_path()
    nk = len(k_line)
    Ek = np.zeros((nk, num_wan))
    factor, R = read_Hr_dat()
    for i in range(nk):
        H = matrix_construct(factor, R, k_line[i][0], k_line[i][1], k_line[i][2])
        E, _ = LA.eigh(H)
        Ek[i, :] = np.sort(np.real(E))
        # print("Process Finished ", i * 1000 / len(k_line), '%')

    plt.plot(Ek)
    plt.show()
    # np.save("data/Ek.npy", Ek - E_fermi)
    print("Process Finished")