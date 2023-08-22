#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于网站的方法提取wannier信息.py    
@Time    :   2023/4/2 9:08
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
    this = ['1', '0', '0']
    print(''.join(this))
    print(''.join(list(map(str, Direct[2]))))
    print(''.join(this) == ''.join(list(map(str, Direct[2]))))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def isinDirect(v_):
    for d in Direct:
        if ''.join(v_) == ''.join(list(map(str, d))):
            return True
    return False


def k_path():
    k_point = []
    for ii in range(len(K_point_path) - 1):
        interval = np.array(K_point_path[ii + 1]) - np.array(K_point_path[ii])
        interval = interval / k_meshes[ii]
        for jj in range(k_meshes[ii]):
            k_point.append(np.array(K_point_path[ii]) + jj * interval)
    return k_point


def read_dat(*args, **kwargs):
    with open("data/wannier90_hr_SVO.dat", "r") as f:
        lines = f.readlines()
    Rij = [[[] for col in range(num_band)] for row in range(num_band)]
    tij = [[[] for col in range(num_band)] for row in range(num_band)]
    for line in lines:
        ll = line.split()
        # ['0', '0', '0', '1', '1', '9.369108', '0.000000']
        if isinDirect(ll[:3]):
            x, y, z, frAt, toAt = list(map(int, ll[:5]))
            t_real, t_image = list(map(float, ll[5:]))
            Rij[frAt-1][toAt-1].append([x, y, z])
            tij[frAt-1][toAt-1].append([t_real + 1j*t_image])
    return Rij, tij


def phase(R1, R2, R3, k1, k2, k3):  # this is exp(ik·R)
    R1_vector = R1 * np.array(basis_vector[0])
    R2_vector = R2 * np.array(basis_vector[1])
    R3_vector = R3 * np.array(basis_vector[2])
    R_vec = R1_vector + R2_vector + R3_vector
    inner_product = np.dot(R_vec, [k1, k2, k3])
    return np.exp(1j * inner_product)


def matrix_construct(tij, Rij, param, param1, param2):
    nb = num_band
    R = Rij
    k1, k2, k3 = param, param1, param2
    h = np.zeros((nb, nb), dtype='complex')
    for i in range(nb):
        for j in range(nb):
            for k in range(len(R[i][j])):
                h[i][j] = h[i][j] + tij[i][j][k][0] * phase(R[i][j][k][0], R[i][j][k][1], R[i][j][k][2], k1, k2, k3)
    return h


if __name__ == '__main__':
    # start here
    num_band = 5
    Direct = [[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 0, 2]]
    rij, factor = read_dat()

    basis_vector = [[1.37287871, 1.37287871, -2.74575742],
                    [-2.74575742, 1.37287871, 1.37287871],
                    [13.36629497, 13.36629497, 13.36629497]]
    E_fermi = -1.3286
    K_point_path = [[0, 0, 0],
                    [0.50000, 0.00000, 0.00000],
                    [0.33333, 0.33333, 0.00000],
                    [0.00000, 0.00000, 0.00000]]
    n = 10
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

    k_line = k_path()
    nk = len(k_line)
    Ek = np.zeros((nk, num_band))

    for i in range(nk):
        H = matrix_construct(factor, rij, k_line[i][0], k_line[i][1], k_line[i][2])
        E, _ = np.linalg.eig(H)
        Ek[i, :] = np.sort(np.real(E))
        # print("Process Finished ", i * 100 / len(k_line), '%')
    # np.save("data/Ek.npy", Ek - E_fermi)
    plt.plot(Ek)
    plt.title('get Ek')
    plt.show()
    print("Process Finished")
