#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   随机测试.py    
@Time    :   2023/2/12 23:42  
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
import yambopy


from JiaxuYan.phonons.TFLG_dynamica_matrix import R_phi


def func_here(*args, **kwargs):
    pass


def dis_atom_a(a_, b_):
    return distance.cdist(a_, b_)


def delta_theta(dis_):
    # dis_ = distance between the interacting atoms from a given atomic configuration corresponding to angle theta
    return 4 * epsilon * (156 * sigma ** 12 / dis_ ** 14 - 42 * sigma ** 6 / dis_ ** 8)


if __name__ == '__main__':
    # start here
    rt3 = 3 ** 0.5
    a = 0.142
    epsilon = 4.6  # meV
    sigma = 0.3276  # nm
    l = np.array([[1, 2], [3, 4]])
    print(list(l*2)[0][1])
    force_constant_my = [[36.50, 24.50, 9.82],
                         [4.037, -3.044, -0.492],
                         [-3.016, 3.948, 0.516],
                         [0.564, 0.129, -0.521],
                         [1.035, 0.166, 0.110]]
    print(np.array(np.arange((9))).reshape((3,3)))
    # b = np.arange(1, 10).reshape(3, 3)
    # H12AAp = np.zeros((3, 3), dtype=complex)
    # for ii in range(3):
    #     H12AAp[ii, ii] = ii+1+1j
    # print(H12AAp)
    # ii = 0
    # hiiBBp = [sv[r] for sv, r in zip(b, range(3))]
    # # v =
    # print(hiiBBp)
    # for index in range(14):  # 一次写四个小矩阵，构建24*24*3的矩阵
    #     index *= 2
    #     i0 = index * 3
    #     i1 = (index + 1) * 3
    #     i2 = (index + 2) * 3
    #     di = {'AA': [i0, i1, i0, i1], 'BB': [i1, i2, i1, i2], 'AB': [i0, i1, i1, i2], 'BA': [i1, i2, i0, i1]}
    #     print(di.values())
    # a_, b_ = np.array([[1], [1], [0]]), np.array([[-1], [-1], [0]])
    # print(distance.cdist([a_[:, 0]], [b_[:, 0]]))
    # print(a_[:, 0] - b_[:, 0])
    # w11 = np.dot(R_phi(0), np.array([1, 0, 1]).reshape(3, 1))
    # print(w11)
    # print(np.arccos(2.5 / 7 ** 0.5))
    # print(np.arctan(3**0.5 / 5))
    # v11 = np.array([1, 1, 0]).reshape(1, 3)
    # v11 = np.c_[v11, 1]
    # Trans = np.array([[1, 0, 0, 0],
    #                   [0, 1, 0, 0],
    #                   [0, 0, 1, 0],
    #                   [-2, -2, -0, 1]])
    # a11 = np.matmul(v11, Trans).reshape(np.size(v11,1), np.size(v11,0))
    # print(a11[:3])
    #
    # c = np.array([1, 1]).reshape(1, 2)
    # c = np.c_[c, 1]  # .reshape(np.size(c, 1)+1, 1)
    # Tr = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [1, 1, 1]])
    # Tr1 = np.array([[1, 0, 0],
    #                 [0, 1, 0],
    #                 [-1, -1, 1]])
    # Tw = R_phi(np.pi/4)
    # print(c)
    # c = np.matmul(c, Tr)
    # print(c)
    # c = np.matmul(c, Tw)
    # print(c)
    # c = np.matmul(c, Tr1)
    # print(c)
    # v11 = np.array([1, 0, 0.335]) * a
    # v12 = np.array([-1, 0.9, 0]) * a
    # v46 = np.array([1 / 2, 3 * rt3 / 2, 0]) * a
    # # -0.011791568962645476
    # dis1 = dis_atom_a([v11], [v12])
    # print(delta_theta(dis1))
    # dis2 = dis_atom_a([v11], [v46])
    # dis = dis_atom_a([v11], [v46])
    #
    # xx = np.linspace(0.1, 0.5, 100)
    # yy = delta_theta(xx)
    # plt.plot(xx, yy)
    # plt.show()
    # a = np.array([[1], [0], [0]])
    # f = np.diag([36.50, 24.50, 9.82, 8.80, -3.23, -0.40, 3.00, -5.25, 0.15, -1.92, 2.29, -0.58])

    # print(a)
    # print(np.size(a))
    # print(np.size(a, 0))
    # print(np.size(a, 1))
    # print(f[:3, :3])
    # print(np.size(f[:3, :3]))
    # a = [np.array([[1, 2, 0.],
    #                [2, 3, 0.],
    #                [0., 0., 4]]),
    #      np.array([[1, 2, 0.],
    #                [2, 3, 0.],
    #                [0., 0., 4]])]
    # print(sum(a))
    # print(sum([sum(a), sum(a)]))
    # c = np.diag([-1, 1, 1])
    # a = np.diag([-1, -1, 1])
    # p = R_phi(np.pi/3)
    # b = np.arange(1, 10).reshape(3, 3)
    # print(np.linalg.inv(np.linalg.inv(b)))
    # print(p@b@np.linalg.inv(p))
    # print(np.linalg.inv(p)@b@p)


    # sp_ = 2 six atoms, A site
    # # 1
    # r_ = R_phi(0) @ Phi_sp_2 @ R_phi(0).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = R_phi(0) @ Phi_up_2 @ R_phi(0).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)
    # # 2
    # # r_ = sigma_y @ R_phi(2 * np.pi / 3) @ Phi_sp_2 @ R_phi(2 * np.pi / 3).T @ np.linalg.inv(sigma_y)
    # # 这文献为什么不转π/3，非要转到下一个，再sigma变换回来
    # r_ = R_phi(1 * np.pi / 3) @ Phi_sp_2 @ R_phi(1 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = r_ = R_phi(1 * np.pi / 3) @ Phi_up_2 @ R_phi(1 * np.pi / 3).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)
    # # 3
    # r_ = R_phi(2 * np.pi / 3) @ Phi_sp_2 @ R_phi(2 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = R_phi(2 * np.pi / 3) @ Phi_up_2 @ R_phi(2 * np.pi / 3).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)
    # # 4
    # r_ = R_phi(3 * np.pi / 3) @ Phi_up_2 @ R_phi(3 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = R_phi(2 * np.pi / 3 * im) @ Phi_up_1 @ R_phi(2 * np.pi / 3 * im).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)
    # # 5
    # r_ = R_phi(4 * np.pi / 3) @ Phi_sp_2 @ R_phi(4 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = R_phi(2 * np.pi / 3 * im) @ Phi_up_1 @ R_phi(2 * np.pi / 3 * im).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)
    # # 6
    # r_ = sigma_y @ R_phi(4 * np.pi / 3) @ Phi_sp_2 @ R_phi(4 * np.pi / 3).T @ np.linalg.inv(sigma_y)
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    #
    # ra_ = R_phi(2 * np.pi / 3 * im) @ Phi_up_1 @ R_phi(2 * np.pi / 3 * im).T
    # kA_.append(ra_)
    # kB_.append(U.T @ ra_ @ U)

    # # sp_ = 4 six atoms, B site, two angle types
    # # 1
    # r_ = R_phi(0) @ Phi_sp_4_up @ R_phi(0).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # # 2
    # r_ = R_phi(0) @ Phi_sp_4_down @ R_phi(0).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # # 3  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # r_ = R_phi(2 * np.pi / 3) @ Phi_sp_4_up @ R_phi(2 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # # 4
    # r_ = R_phi(2 * np.pi / 3) @ Phi_sp_4_down@ R_phi(2 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # # 5
    # r_ = R_phi(4 * np.pi / 3) @ Phi_sp_4_up @ R_phi(4 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # # 6
    # r_ = R_phi(4 * np.pi / 3) @ Phi_sp_4_down @ R_phi(4 * np.pi / 3).T
    # kA.append(r_)
    # kB.append(U.T @ r_ @ U)
    # index *= 2
    # i0 = index * 3
    # i1 = (index + 1) * 3
    # i2 = (index + 2) * 3
    # # di = {'AA': [i0, i1, i0, i1], 'BB': [i1, i2, i1, i2], 'AB': [i0, i1, i1, i2], 'BA': [i1, i2, i0, i1]}
    # D[i0:i1, i0:i1] = sum([sum(KAB1), sum(KAA2), sum(KAB3), sum(KAB4)]) - DAAf
    # D[i1:i2, i1:i2] = sum([sum(KBA1), sum(KBB2), sum(KBA3), sum(KBA4)]) - DBBf
    # D[i0:i1, i1:i2] = -DABf
    # D[i1:i2, i0:i1] = -np.conj(DABs.T)  # -DBAf  # np.conj(DABs.T)

    # sp_ = 1 three atoms, B site
    # hij vector j=[i for i in range(1, 7)], vij = vector i2j,这个文献他Fig2按顺时针转,这样角度就毫无顺序了就
