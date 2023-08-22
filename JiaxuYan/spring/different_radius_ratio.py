#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   different_radius_ratio.py
@Time    :   2022/11/16 17:52  
@E-mail  :   iamwxyoung@qq.com
@Tips    :  different_radius_ratio.py
    # x = np.arange(6)
    # y = np.arange(6) ** 2
    # xy = np.stack([x, y], axis=1)
    # segments = np.concatenate((xy, y.reshape(6, 1)), axis=1)
    # 不同维度数组合并
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
import numpy as np
import math
import os

from scipy.spatial import distance


class Circle:
    x = 0.0
    y = 0.0
    r = 1.0

    def __init__(self, x, y, R):
        self.x = float(x)
        self.y = float(y)
        self.r = float(R)

    def calArea(self):
        return np.pi * self.r ** 2


def matrix_transformation(x_, y_, theta):
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    xT, yT = [], []
    for k, v in zip(x_, y_):
        twistMatrix = np.dot([k, v], Matrix)
        # 矩阵 1X2 * 2X2 = 1X2
        xT.append(twistMatrix[0])
        yT.append(twistMatrix[1])
    return np.array(xT), np.array(yT)


# 求两圆相交的面积
def calShadow(circle1, circle2):
    d = ((circle1.x - circle2.x) ** 2 + (circle1.y - circle2.y) ** 2) ** 0.5
    if d == 0:
        return circle1.calArea()
    if d > 14:
        print('[', circle1.x, ',', circle1.y, '] 和 [', circle2.x, ',', circle2.y, ']不重叠')
        return 0.0
    else:
        ang1 = np.arccos((circle1.r ** 2 + d ** 2 - circle2.r ** 2) / 2.0 / circle1.r / d)
        ang2 = np.arccos((-circle1.r ** 2 + d ** 2 + circle2.r ** 2) / 2.0 / circle2.r / d)
        area = ang1 * circle1.r ** 2 + ang2 * circle2.r ** 2 - d * circle1.r * np.sin(ang1)
        return area


def over_flow_drop(xL, yL, zL, R):
    # 减小计算量，保证x y z 删除原子数一致
    xDrop = np.delete(xL, np.where(yL.__abs__() > R))  #
    yDrop = np.delete(yL, np.where(yL.__abs__() > R))
    zDrop = np.delete(zL, np.where(yL.__abs__() > R))
    return xDrop, yDrop, zDrop


# 计算(x,y)和原点(0,0)的距离
def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def genGraphene(Super=10):  # 返回新的坐标的大胞
    # 原胞中C的坐标
    a = (2.460, 0, 0)
    b = (2.460 / 2, 2.460 / 2 * math.sqrt(3), 0)
    c = (0, 0, 10)
    # 扩胞矩阵
    super_x = Super
    super_y = Super
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
    allAtoms = []
    for i in range(super_x):
        for j in range(super_y):
            newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.5]
            newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.5]
            allAtoms.append(newC1)
            allAtoms.append(newC2)
    newAllAtoms = np.dot(np.array(allAtoms), extendLattice)

    # with open('data/graphene.data', 'w') as writer:
    #     writer.write('# MoS2 By Celeste\n\n')
    #     writer.write('%d atoms\n' % (2 * Super * Super))
    #     writer.write('1 atom types\n\n')
    #     writer.write('%7.3f %7.3f xlo xhi\n' % (a[1], a[0]))
    #     writer.write('%7.3f %7.3f ylo yhi\n' % (a[1], b[1]))
    #     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, c[2]))
    #     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (a[0]/2, 0.0, 0.0))
    #     writer.write('  Masses\n\n')
    #     writer.write('1 1.2011\n\n')
    #     writer.write('Atoms\n\n')
    #
    #     index = 1
    #     for xl, yl in zip(x_List, y_List):
    #         writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, xl, yl, 10))
    #         index += 1
    x_mean = np.mean(newAllAtoms[:, 0])
    y_mean = np.mean(newAllAtoms[:, 1])
    xList = newAllAtoms[:, 0] - x_mean
    yList = newAllAtoms[:, 1] - y_mean
    zList = newAllAtoms[:, 2]

    x_drop, y_drop, z_drop = over_flow_drop(xList, yList, zList, x_mean)

    norm_xy_inequation = (normXY(x_drop, y_drop) > y_mean)

    norm_x_ = np.delete(x_drop, np.where(norm_xy_inequation))
    norm_y_ = np.delete(y_drop, np.where(norm_xy_inequation))
    norm_z_ = np.delete(z_drop, np.where(norm_xy_inequation))

    return norm_x_, norm_y_, norm_z_


def cal_total_area(s_1, s_2):
    circle = Circle(0, 0, a_cc).calArea()
    total_s1 = len(s_1) * circle

    return total_s1


def sum_s1_s2_area(s_1, s_2):
    empty_list = []
    for fl in s_1:  # first layer atoms
        min_distance = distance.cdist([fl], s_2, 'euclidean').min(axis=1)
        index_tuple = np.where(distance.cdist(s_2, [fl], 'euclidean') == min_distance)
        atom_area = calShadow(Circle(fl[0], fl[1], a_cc),
                              Circle(s_2[index_tuple[0]][0][0], s_2[index_tuple[0]][0][1], a_cc))
        empty_list.append(atom_area)

    return sum(empty_list)


def cal_euclidean_s1_s2_(s_1, s_2):
    dis1 = distance.cdist(s_1, s_2, 'euclidean').min(axis=1)
    dis2 = distance.cdist(s_1, s_2, 'euclidean').min(axis=0)
    # 先取三分之一acc吧
    index_s1 = np.where(dis1 <= a_cc)
    index_s2 = np.where(dis2 <= a_cc)

    return index_s1, index_s2


def twist_bilayer_graphene(*args, **kwargs):
    [x_, y_, z_] = [args[i] for i in range(3)]
    layer_1 = np.stack([x_, y_, z_], axis=1)

    print(layer_1)


if __name__ == '__main__':
    pass
    # start here
    # time1 = time.time()
    Super = 80
    sub_value = 2
    a_cc = 1.42 / sub_value
    norm_x, norm_y, norm_z = genGraphene(Super=Super)

    layer_init = np.stack([norm_x, norm_y, norm_z], axis=1)

    # for i in range(31):
    #     print(i)
    #     angle = i
    #     theta = np.deg2rad(angle)
    #     twist_x, twist_y = matrix_transformation(norm_x, norm_y, theta=theta)
    #     layer_twist = np.stack([twist_x, twist_y, norm_z], axis=1)
    #     index_S1, index_S2 = cal_euclidean_s1_s2_(layer_init[:, :2], layer_twist[:, :2])
    #     tmp_init = layer_init[index_S1]
    #     tmp_twist = layer_twist[index_S2]
    #
    #     overlap_area = sum_s1_s2_area(s_1=tmp_init, s_2=tmp_twist)
    #     all_area = cal_total_area(s_1=layer_init, s_2=layer_twist)
    #     fig_size = 10
    #     plt.figure(figsize=(fig_size, fig_size), dpi=200)
    #     plt.title("angle = %.2f° " % angle)
    #     plt.scatter(layer_init[:, 0], layer_init[:, 1], 3, color='g')
    #     plt.scatter(layer_twist[:, 0], layer_twist[:, 1], 3, color='b')
    #
    #     # plt.scatter(tmp_init[:, 0], tmp_init[:, 1], 3)
    #     # plt.scatter(tmp_twist[:, 0], tmp_twist[:, 1], 3)
    #     plt.savefig('png/angle_%d.png' % i)
    #     # plt.show()

    # 重叠的原子找出来了，原子的半径按多少计算？acc/3
    # xd = xlwt.Workbook()
    # sheet1 = xd.add_sheet('sheet1')
    # title = ["angle", "ratio", "percent"]
    # row = 0
    # col = 0
    # for i in title:
    #     sheet1.write(row, col, i, style=xlwt.easyxf('font: bold on'))
    #     col += 1
    # range_step = 6000
    # for i in range(1):
    #     print(".", end="")
    #     if i % 50 == 0:
    #         print(' %d / %d' % (i, range_step - 1), end="\n")
    #     row += 1
    #     col = 0
    #     angle = i
    #     theta = np.deg2rad(angle)
    #     twist_x, twist_y = matrix_transformation(norm_x, norm_y, theta=theta)
    #     layer_twist = np.stack([twist_x, twist_y, norm_z], axis=1)
    #     index_S1, index_S2 = cal_euclidean_s1_s2_(layer_init[:, :2], layer_twist[:, :2])
    #     tmp_init = layer_init[index_S1]
    #     tmp_twist = layer_twist[index_S2]
    #
    #     overlap_area = sum_s1_s2_area(s_1=tmp_init, s_2=tmp_twist)
    #     all_area = cal_total_area(s_1=layer_init, s_2=layer_twist)
    #     ratio = overlap_area / all_area
    #     sheet1.write(row, 0, angle)
    #     sheet1.write(row, 1, ratio)
    #     sheet1.write(row, 2, ratio*100)
    # if os.path.exists("data/ratio_bigger_acc.xls"):
    #     os.remove("data/ratio_bigger_acc.xls")
    # xd.save("data/ratio_bigger_acc.xls")
    # print(all_area, overlap_area)
    # for ti in tmp_init:  # first layer atoms
    #     minDistance = distance.cdist([ti], tmp_twist, 'euclidean').min(axis=1)
    #     indexTuple = np.where(distance.cdist(tmp_twist, [ti], 'euclidean') == minDistance)
    #     result = calShadow(Circle(ti[0], ti[1], a_cc / 3),
    #                        Circle(tmp_twist[indexTuple[0]][0][0], tmp_twist[indexTuple[0]][0][1], a_cc / 3))
    #
    #     print(result)
    #     break

    # t = time.time() - time1
    # print('>> Finished, use time %d s (%.2f min).' % (t, t / 60.0))
