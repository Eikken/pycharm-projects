#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   产生AA_AB堆叠对比信息.py    
@Time    :   2023/1/28 19:31  
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
import math


# 计算(x,y)和原点(0,0)的距离
from scipy.spatial import distance


def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def generate_AA_AB_stack(*args, **kwargs):

    sup = kwargs['Super']
    # 原胞中C的坐标
    a_ = 2.460
    a = (a_, 0, 0)
    b = (a_ / 2, a_ / 2 * math.sqrt(3), 0)
    c = (0, 0, 10)
    # 扩胞矩阵
    super_x = sup
    super_y = sup
    super_z = 1

    extendCellMatrix = np.array([[super_x, 0, 0],
                                 [0, super_y, 0],
                                 [0, 0, super_z]])
    lattice = np.array([a, b, c])
    # 矩阵右乘扩胞矩阵3X3 * 3X3，生成新的大胞
    extendLattice = np.dot(lattice, extendCellMatrix)
    # C1 = [0, 0, 0.5]
    # C2 = [1 / float(3), 1 / float(3), 0.5]

    Frac1 = 0
    Frac2 = 1 / float(3)
    Frac3 = 2 / float(3)

    if args[0] == 'AA':
        layer_1_atoms = []
        layer_2_atoms = []
        for i in range(super_x):
            for j in range(super_y):
                newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.25]
                newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.25]

                newC3 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.75]
                newC4 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.75]
                # AA stack layer1 and layer2
                layer_1_atoms.append(newC1)
                layer_1_atoms.append(newC2)

                layer_2_atoms.append(newC3)
                layer_2_atoms.append(newC4)

        all_layer1 = np.dot(np.array(layer_1_atoms), extendLattice)
        all_layer2 = np.dot(np.array(layer_2_atoms), extendLattice)

        x_mean_1, y_mean_1 = np.mean(all_layer1[:, 0]), np.mean(all_layer1[:, 1])
        x_list_1, y_list_1, z_list_1 = all_layer1[:, 0] - x_mean_1, all_layer1[:, 1] - y_mean_1, all_layer1[:, 2]
        norm_xy_inequation_1 = (normXY(x_list_1, y_list_1, ) > y_mean_1)
        norm_x_1 = np.delete(x_list_1, np.where(norm_xy_inequation_1))
        norm_y_1 = np.delete(y_list_1, np.where(norm_xy_inequation_1))
        norm_z_1 = np.delete(z_list_1, np.where(norm_xy_inequation_1))

        x_mean_2, y_mean_2 = np.mean(all_layer2[:, 0]), np.mean(all_layer2[:, 1])
        x_list_2, y_list_2, z_list_2 = all_layer2[:, 0] - x_mean_2, all_layer2[:, 1] - y_mean_2, all_layer2[:, 2]
        norm_xy_inequation_2 = (normXY(x_list_2, y_list_2) > y_mean_2)
        norm_x_2 = np.delete(x_list_2, np.where(norm_xy_inequation_2))
        norm_y_2 = np.delete(y_list_2, np.where(norm_xy_inequation_2))
        norm_z_2 = np.delete(z_list_2, np.where(norm_xy_inequation_2))

        AA_layer1 = np.stack([norm_x_1, norm_y_1, norm_z_1], axis=1)
        AA_layer2 = np.stack([norm_x_2, norm_y_2, norm_z_2], axis=1)

        return AA_layer1, AA_layer2

    if args[0] == 'AB':
        layer_1_atoms = []
        layer_2_atoms = []
        for i in range(super_x):
            for j in range(super_y):
                newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.25]
                newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.75]
                layer_1_atoms.append(newC1)
                layer_2_atoms.append(newC2)

                newC3 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.25]
                newC4 = [(Frac3 + i) / super_x, (Frac3 + j) / super_y, 0.75]
                # AB stack layer1 and layer2
                layer_1_atoms.append(newC3)
                layer_2_atoms.append(newC4)

        all_layer1 = np.dot(np.array(layer_1_atoms), extendLattice)
        all_layer2 = np.dot(np.array(layer_2_atoms), extendLattice)
        x_mean_1, y_mean_1 = np.mean(all_layer1[:, 0]), np.mean(all_layer1[:, 1])

        x_list_1, y_list_1, z_list_1 = all_layer1[:, 0] - x_mean_1, all_layer1[:, 1] - y_mean_1, all_layer1[:, 2]
        norm_xy_inequation_1 = (normXY(x_list_1, y_list_1,) > y_mean_1)
        norm_x_1 = np.delete(x_list_1, np.where(norm_xy_inequation_1))
        norm_y_1 = np.delete(y_list_1, np.where(norm_xy_inequation_1))
        norm_z_1 = np.delete(z_list_1, np.where(norm_xy_inequation_1))

        x_mean_2, y_mean_2 = np.mean(all_layer2[:, 0]), np.mean(all_layer2[:, 1])
        x_list_2, y_list_2, z_list_2 = all_layer2[:, 0] - x_mean_2, all_layer2[:, 1] - y_mean_2 - a_/3**0.5, all_layer2[:, 2]
        norm_xy_inequation_2 = (normXY(x_list_2, y_list_2) > y_mean_2)
        norm_x_2 = np.delete(x_list_2, np.where(norm_xy_inequation_2))
        norm_y_2 = np.delete(y_list_2, np.where(norm_xy_inequation_2))
        norm_z_2 = np.delete(z_list_2, np.where(norm_xy_inequation_2))

        AB_layer1 = np.stack([norm_x_1, norm_y_1, norm_z_1], axis=1)
        AB_layer2 = np.stack([norm_x_2, norm_y_2, norm_z_2], axis=1)

        return AB_layer1, AB_layer2


def matrix_transformation(this_layer, theta_):
    Matrix = np.array([
        [np.cos(theta_), -np.sin(theta_)],
        [np.sin(theta_), np.cos(theta_)]
    ])
    xT, yT = [], []
    for k, v in zip(this_layer[:, 0], this_layer[:, 1]):
        twistMatrix = np.dot([k, v], Matrix)
        # 矩阵 1X2 * 2X2 = 1X2
        xT.append(twistMatrix[0])
        yT.append(twistMatrix[1])

    return np.stack([xT, yT, this_layer[:, 2]], axis=1)


def cal_euclidean_s1_s2_(s_1, s_2):
    dis1 = distance.cdist(s_1, s_2, 'euclidean').min(axis=1)
    dis2 = distance.cdist(s_1, s_2, 'euclidean').min(axis=0)
    # 先取三分之一acc吧
    index_s1 = np.where(dis1 <= 0.01)
    index_s2 = np.where(dis2 <= 0.01)

    return index_s1, index_s2


if __name__ == '__main__':
    # start here
    Super = 50
    a__ = 2.46
    a_cc = a__/3**0.5
    AA_layer_1, AA_layer_2 = generate_AA_AB_stack('AB', Super=Super)
    # AB_layer_1, AB_layer_2 = generate_AA_AB_stack('AB', Super=Super)

    angle_list = [4.41, 5.09, 6.01, 7.34, 8.61, 9.43, 10.42, 11.64, 13.17, 15.18, 16.43, 17.90, 19.65, 21.79, 24.43,
                  26.01, 27.8, 18.73, 19.03, 19.27, 19.65, 20.32, 23.48, 24.02, 25.04, 25.46, 28.78, 29.41]

    plt.figure(figsize=(10, 10))
    for i in sorted(angle_list):
        angle = i
        print("angle", angle)
        theta = np.deg2rad(angle)
        twist_layer = matrix_transformation(AA_layer_2, theta_=theta)
        index_S1, index_S2 = cal_euclidean_s1_s2_(AA_layer_1[:, :2], twist_layer[:, :2])
        tmp_init = AA_layer_1[index_S1]
        tmp_twist = twist_layer[index_S2]
        plt.scatter([0], [0], 30, marker='*', color='black')
        plt.scatter(AA_layer_1[:, 0], AA_layer_1[:, 1], 5, color='b')
        plt.scatter(twist_layer[:, 0], twist_layer[:, 1], 5, color='g')
        plt.scatter(tmp_init[:, 0], tmp_init[:, 1], 30, color='r')
        # plt.scatter(tmp_twist[:, 0], tmp_twist[:, 1], 30, color='y')
        plt.xticks([])
        plt.yticks([])
        plt.title('AB stack %.2f°'%i)
        # plt.show()
        plt.savefig('png/ABshow/angle_%.2f.png' % i)
        plt.clf()
        #     # plt.show()
        # break
    # plt.figure(figsize=(5, 5))
    # plt.scatter([0], [0], 30, marker='*', color='black')
    # plt.scatter(AA_layer_1[:, 0], AA_layer_1[:, 1], 20, color='b')
    # plt.scatter(AA_layer_2[:, 0], AA_layer_2[:, 1], 5, color='g')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('AA stack')
    # plt.show()
    #
    # plt.figure(figsize=(5, 5))
    # plt.scatter([0], [0], marker='*', color='black')
    # plt.scatter(AB_layer_1[:, 0], AB_layer_1[:, 1], 20, color='b')
    # plt.scatter(AB_layer_2[:, 0], AB_layer_2[:, 1], 5, color='r')
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('AB stack')
    # plt.show()
