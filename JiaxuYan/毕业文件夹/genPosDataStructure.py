#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   genPosDataStructure.py    
@Time    :   2022/8/3 11:47  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   排列组合计算、欧式距离计算
            a_cc_NN = 1.42097
            a_cc_NNN = 2.46120
'''

import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
from itertools import product
from itertools import combinations
from scipy.spatial import distance

from JiaxuYan.hamilton import my_twist_constants
import random


def randomcolor():
    # colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    colorArr = ['red','yellow','green','blue']
    # color = ""
    # for i in range(6):
    #     color += colorArr[random.randint(0,14)]
    color = colorArr[random.randint(0, 3)]
    return color


@pb.onsite_energy_modifier
def potential(x, y):
    return np.sin(x)**2 + np.cos(y)**2


def gen_lattice():
    acc_NN = my_twist_constants.a_nn__21_7
    acc_NNN = my_twist_constants.a_nnn_21_7
    acc_T = my_twist_constants.a_t_21_7 # 可以修改的参数值

    a_ = my_twist_constants.a_lat_21_7

    rf = pd.read_csv(r'D:\Celeste\PycharmProjects\JiaxuYan\pbtest\posdata\1.xyz')
    dataSet = rf.values
    lenAtoms = int(rf.columns[0])

    r2 = [-1 / 2 * a_, -3 ** 0.5 / 2 * a_, 0]
    r1 = [1 * a_, 0, 0]

    lat = pb.Lattice(a1=r1, a2=r2)
    atom_dict = {}
    for i in range(1, lenAtoms + 1):
        cpos = [float(j) for j in dataSet[i][0].split(' ')[1:] if j != ''][:3]  # 笛卡尔坐标
        lat.add_sublattices(
            ('C' + str(i), [j for j in cpos]),
        )
        atom_dict['C' + str(i)] = [j for j in cpos]  # atom & position information {'C1':[x, y, z], et.}

    atom_keys = list(atom_dict.keys())  # atoms' keys like [C1, C2, C3, .et]
    # should be product C(2/28) then calculate distance
    # or calculate the nearest distance of all atoms find the pair atoms?
    # 先C(2/28)排列组合，然后计算(Ci,Cj)的distance, 等于a_cc就break，就等于找到了NN的点
    lat.register_hopping_energies({
        'gamma0': -2.8,  # [eV] intralayer
        'gamma1': -0.4,  # [eV] interlayer
    })
    ca_list = list(combinations(atom_keys, 2))

    acc_NN_list = []  # 最近邻Cij mapping list 集合
    acc_NNN_list = []  # 次近邻，暂时未使用
    acc_T_list = []  # 垂直mapping Cij
    acc_NN_a1 = [[], []]  # T/F
    acc_NN_a2 = [[], []]  # T/F
    acc_NN_a3 = [[], []]  # T/F

    esm = 0.01  # 衡量距离绝对值差值

    keyIndex = 0
    for i in range(1, lenAtoms):
        this_i = lenAtoms - i
        for j in range(this_i):
            (Ci, Cj) = ca_list[keyIndex]
            pos_i, pos_j = atom_dict[Ci], atom_dict[Cj]
            dis_ij = distance.cdist(np.array([pos_i]), np.array([pos_j]), 'euclidean')
            # dis_ij mapping的是Cij胞内accNN
            dis_ija1_T = distance.cdist(np.array([pos_i]),
                                        np.array([[px + py for px, py in zip(pos_j, r1)]]), 'euclidean')
            dis_ija2_T = distance.cdist(np.array([pos_i]),
                                        np.array([[px + py for px, py in zip(pos_j, r2)]]), 'euclidean')
            dis_ija3_T = distance.cdist(np.array([pos_i]),
                                        np.array([[px + py for px, py in zip(pos_j,
                                                                             [r1_ + r2_ for r1_, r2_ in
                                                                              zip(r1, r2)])]]), 'euclidean')

            # 这个distance a1 a2必须用三维向量，二维向量会产生不可估测的一些误差
            dis_ija1_F = distance.cdist(np.array([pos_j]),
                                        np.array([[px + py for px, py in zip(pos_i, r1)]]), 'euclidean')
            dis_ija2_F = distance.cdist(np.array([pos_j]),
                                        np.array([[px + py for px, py in zip(pos_i, r2)]]), 'euclidean')
            dis_ija3_F = distance.cdist(np.array([pos_j]),
                                        np.array([[px + py for px, py in zip(pos_i,
                                                                             [r1_ + r2_ for r1_, r2_ in
                                                                              zip(r1, r2)])]]), 'euclidean')

            # here a1, a2 as direction vector
            # a1_T/F, a2_T/F as conjunction[1, 0], [-1, 0], [0, -1], [0, 1] bind relationship.
            if np.abs(dis_ij - acc_NN) < esm:
                acc_NN_list.append(ca_list[keyIndex])
                keyIndex += 1
                continue
                # 为避免产生不必要的误差，
                # print(ca_list[keyIndex], dis_ij)
            elif np.abs(dis_ij - acc_NNN) < esm:
                acc_NNN_list.append(ca_list[keyIndex])
                keyIndex += 1
                continue
                # print(ca_list[keyIndex], dis_ij)
            elif np.abs(dis_ij - acc_T) < esm:
                acc_T_list.append(ca_list[keyIndex])
                keyIndex += 1
                continue
                # print(ca_list[keyIndex], dis_ij)

            if np.abs(dis_ija1_T - acc_NN) < esm:
                acc_NN_a1[0].append(ca_list[keyIndex])
                # print('a1T', ca_list[keyIndex], dis_ija1_T)
                # 在可视范围内a1T保存了layer1和layer2的[-1, 0]
                keyIndex += 1
                continue

            if np.abs(dis_ija1_F - acc_NN) < esm:
                acc_NN_a1[1].append(ca_list[keyIndex])
                # print('a1F', ca_list[keyIndex], dis_ija1_F)
                # 在可视范围内a1F保存了layer1和layer2的[-1, 0]
                keyIndex += 1
                continue

            if np.abs(dis_ija2_T - acc_NN) < esm:
                acc_NN_a2[0].append(ca_list[keyIndex])
                # print('a2T', ca_list[keyIndex], dis_ija2_T)
                # 在可视范围内a2T为空
                keyIndex += 1
                continue

            if np.abs(dis_ija2_F - acc_NN) < esm:
                acc_NN_a2[1].append(ca_list[keyIndex])
                # print('a2F', ca_list[keyIndex], dis_ija2_F)
                # 在可视范围内a2F保存了layer1和layer2的[0, 1]
                keyIndex += 1
                continue

            if np.abs(dis_ija3_T - acc_NN) < esm:
                acc_NN_a3[1].append(ca_list[keyIndex])
                # print('a3T', ca_list[keyIndex], dis_ija3_T)
                # 在可视范围内a1F保存了layer1和layer2的[-1, 0]
                keyIndex += 1
                continue

            if np.abs(dis_ija3_F - acc_NN) < esm:
                acc_NN_a3[1].append(ca_list[keyIndex])
                # print('a3F', ca_list[keyIndex], dis_ija3_F)
                # 在可视范围内a2F保存了layer1和layer2的[0, 1]
                keyIndex += 1
                continue

            keyIndex += 1

    # ([0, 0], Ci, Cj, gamma0) is finish.
    # [1, 0]、[0, -1]、[1, -1] conj [-1, 0]、[0, 1]、[-1, 1]
    # lattice is a_, mapping 下一个 supercell a1方向将x向右平移y不变，a2方向将x向左平移y向下平移
    # 在distance.cdist()中Cj代入平移后的坐标计算即可。

    hoping_energy = 1
    for item in acc_T_list:
        lat.add_hoppings(([0, 0], item[0], item[1], 'gamma1'))

    for item in acc_NN_list:
        lat.add_hoppings(([0, 0], item[0], item[1], 'gamma0'))
    ###########################################################################################
    # for item1, item2 in zip(acc_NN_a1[0], acc_NN_a1[1]):  # 共轭添加
    #     # ######## 此处 for 循环产生了一个预期之外的错误，记得解决 ########
    #     print('a1 >> ', item1, item2)
    #     lat.add_hoppings(([-hoping_energy, 0], item1[0], item1[1], 'gamma0'))
    #     lat.add_hoppings(([0, hoping_energy], item2[0], item2[1], 'gamma0'))
    #
    # for item1, item2 in zip(acc_NN_a2[0], acc_NN_a2[1]):
    #     print('a2 >> ', item1, item2)
    #     lat.add_hoppings(([-hoping_energy, 0], item1[0], item1[1], 'gamma0'))
    #     lat.add_hoppings(([0, hoping_energy], item2[0], item2[1], 'gamma0'))
    ##########################################################################################
    # 重写item1 item2产生的错误，该错误是由传递方向不严谨导致，除a1,a2外还有a1+a2方向 as below >>
    # 共轭添加,一共添加4次,包括[+-1, 0]和[0, +-1]
    # 此处的123还是132顺序很重要！important!
    acc_NN_a123 = acc_NN_a1 + acc_NN_a2 + acc_NN_a3 + [acc_NN_list] + [acc_T_list]  # [a1T, a1F, a2T, a2F]
    for i in range(6):
        if i == 0:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([hoping_energy, 0], item1[0], item1[1], 'gamma0'))
        elif i == 1:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([-hoping_energy, 0], item1[0], item1[1], 'gamma0'))
        elif i == 2:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([0, hoping_energy], item1[0], item1[1], 'gamma0'))
        elif i == 3:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([0, -hoping_energy], item1[0], item1[1], 'gamma0'))
        elif i == 4:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([hoping_energy, hoping_energy], item1[0], item1[1], 'gamma0'))
        elif i == 5:
            for item1 in acc_NN_a123[i]:
                lat.add_hoppings(([-hoping_energy, -hoping_energy], item1[0], item1[1], 'gamma0'))

    lat.min_neighbors = 2
    return lat, acc_NN_a123


if __name__ == '__main__':
    c0 = 0.335  # [nm] graphene interlayer spacing
    lattice, all_hopping_acc = gen_lattice()
    pb.pltutils.use_style()
    # lattice.plot()
    # plt.title('lattice')
    # plt.show()

    # lattice.plot_brillouin_zone()
    # plt.title('brillouin')
    # plt.show()
    #
    shape = pb.circle(radius=15)
    model = pb.Model(lattice,
                     # shape,
                     pb.translational_symmetry()
                     )
    plt.figure(figsize=(6, 5), dpi=200)
    model.plot()
    ax = plt.gca()  # 获取边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    fontSz = 18
    xk = [-10, -5, 0, 5, 10]
    plt.xticks(xk, xk)
    plt.xlim(-10, 13)
    plt.ylim(-13, 7)
    #

    plt.title('model', fontsize=fontSz)
    plt.ylabel('y (nm)', fontsize=fontSz)
    plt.xlabel('x (nm)', fontsize=fontSz)
    plt.tick_params(labelsize=fontSz)
    plt.show()
    # #
    # model = pb.Model(lattice,
    #                  pb.translational_symmetry(),
    #                  # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
    #                  shape,
    #                  )
    # solver = pb.solver.lapack(model)
    # a_cc = 6.51172 / 3**0.5  # 1.42098
    # # print(a_cc)
    # Gamma = [0, 0]
    # K1 = [-4 * np.pi / (3 * 3 ** 0.5 * a_cc), 0]
    # M = [0, 2 * np.pi / (3 * a_cc)]
    # K2 = [2 * np.pi / (3 * 3 ** 0.5 * a_cc), 2 * np.pi / (3 * a_cc)]
    #
    # bands = solver.calc_bands(Gamma, K2, M, Gamma, step=0.001)
    # # bands = solver.calc_bands(K1, Gamma, M, K2, step=0.001)
    # plt.figure(figsize=(6, 5), dpi=200)
    # # bands.plot(point_labels=[r'$\Gamma$', 'K', 'M', r'$\Gamma$'], lw=0.75)
    # kp = bands.k_path
    # eg = bands.energy
    # xkp = np.arange(len(kp))
    # for i in range(28):
    #     if i % 2 == 0:
    #         plt.plot(xkp, eg[:, i], lw=1, color='blue')
    #     else:
    #         plt.plot(xkp, eg[:, i], lw=1, color='green')
    #
    # hh = 5.5
    # plt.ylim(-hh, hh)
    # # plt.margins(x=0)
    # plt.xlim(0, len(kp))
    fontSz = 18
    # xk = [0, 644, 966, 1521]
    # plt.xticks(xk, [r'$\Gamma$', 'K', 'M', r'$\Gamma$'])
    # plt.axvline(644, color='gray', linestyle='--', linewidth=1, label='K')
    # plt.axvline(966, color='gray', linestyle='--', linewidth=1, label='K')
    # plt.title('band', fontsize=fontSz)
    # plt.ylabel('Energy (eV)', fontsize=fontSz)
    # # plt.xlabel('k-space', fontsize=fontSz)
    # plt.tick_params(labelsize=fontSz)
    # plt.show()

    # model = pb.Model(lattice,
    #                  pb.translational_symmetry())
    # a_cc = 6.51172 / 3**0.5  # 1.42098
    # Gamma = [0, 0]
    # K1 = [-4 * np.pi / (3 * 3 ** 0.5 * a_cc), 0]
    # M = [0, 2 * np.pi / (3 * a_cc)]
    # K2 = [2 * np.pi / (3 * 3 ** 0.5 * a_cc), 2 * np.pi / (3 * a_cc)]
    # solver = pb.solver.lapack(model)
    # bands = solver.calc_bands(Gamma, K2, M, Gamma, step=0.001)
    # # bands = solver.calc_bands(K1, Gamma, M, K2, step=0.001)
    # plt.figure(figsize=(5, 4), dpi=200)
    # bands.plot(point_labels=[r'$\Gamma$', 'K', 'M', r'$\Gamma$'], lw=0.75)
    # hh = 5
    # plt.ylim(-hh, hh)
    # plt.title('supercell plot')
    # plt.show()

    # shape1 = pb.rectangle(1000, 1000)
    # shape2 = pb.circle(radius=500)
    # model = pb.Model(lattice,
    #                  shape1)
    # kpm = pb.kpm(model)
    # ldos = kpm.calc_ldos(energy=np.linspace(-9, 9, 200), broadening=0.17, position=[0, 0])
    # ldos.plot()
    # plt.show()

    # dos = kpm.calc_dos(energy=np.linspace(-9, 9, 500), broadening=0.17, num_random=64)
    # plt.figure(figsize=(6, 5), dpi=200)
    # # dos.plot()
    # # ldos.plot()
    # ax = plt.gca()  # 获取边框
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    # # xk = [0, 20000, 40000, 60000, 80000, 100000]
    # # plt.yticks(xk, [0, 0.2, 0.4, 0.6, 0.8, 1])
    # # xk = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12]
    # # plt.yticks(xk, [0, 0.2, 0.4, 0.6, 0.8, 1])
    # # plt.ylim(0, 110000)
    # plt.xlabel('Energy (eV)', fontsize=fontSz)
    # plt.ylabel('LDOS', fontsize=fontSz)
    # plt.tick_params(labelsize=fontSz)
    #
    # plt.show()
    # print('finish')
