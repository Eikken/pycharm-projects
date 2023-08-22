#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   genMoS2Basic.py    
@Time    :   2022/10/27 9:45  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   

'''
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import xlwt
from scipy.spatial import distance
import pybinding as pb


def over_flow_drop(xL, yL, zL, R):
    # 减小计算量，保证x y z 删除原子数一致
    xDrop = np.delete(xL, np.where(yL.__abs__() > R))  #
    yDrop = np.delete(yL, np.where(yL.__abs__() > R))
    zDrop = np.delete(zL, np.where(yL.__abs__() > R))
    return xDrop, yDrop, zDrop


# 计算(x,y)和原点(0,0)的距离
def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def gen_mos2_lattice(*args, **kwargs):
    # 我得封装一个读取cif文件就能构建lattice的脚本
    a_ = 3.1903
    rt3_ = 3 ** 0.5
    r1_ = [rt3_ / 2 * a_, -1 / 2 * a_, 0]
    r2_ = [0, 1 * a_, 0]
    r3_ = [0, 0, 14.8790]

    super_x = extend
    super_y = extend
    super_z = 1

    extend_matrix = np.array([[super_x, 0, 0],
                              [0, super_y, 0],
                              [0, 0, super_z]])
    basic_lattice = np.array([r1_, r2_, r3_])

    extend_lattice = np.dot(basic_lattice, extend_matrix)

    # print(extend_lattice)
    frac_1 = 1 / float(3)
    frac_2 = 2 / float(3)
    all_atoms = []

    for i in range(super_x):
        for j in range(super_y):
            _s1_up = [(frac_2 + i) / super_x, (frac_1 + j) / super_y, 0.35517]
            _s1_down = [(frac_2 + i) / super_x, (frac_1 + j) / super_y, 0.14483]
            _mo1_mid = [(frac_1 + i) / super_x, (frac_2 + j) / super_y, 0.25]

            # # AA stack
            # _s2_up = [(frac_2 + i) / super_x, (frac_1 + j) / super_y, 0.85517]
            # _s2_down = [(frac_2 + i) / super_x, (frac_1 + j) / super_y, 0.64483]
            # _mo2_mid = [(frac_1 + i) / super_x, (frac_2 + j) / super_y, 0.75]

            # AB stack
            _s2_up = [(frac_1 + i) / super_x, (frac_2 + j) / super_y, 0.85517]
            _s2_down = [(frac_1 + i) / super_x, (frac_2 + j) / super_y, 0.64483]
            _mo2_mid = [(frac_2 + i) / super_x, (frac_1 + j) / super_y, 0.75]

            for atom in [_s1_up, _s1_down, _mo1_mid, _s2_up, _s2_down, _mo2_mid]:
                all_atoms.append(atom)

    new_atoms_ = np.dot(np.array(all_atoms), extend_lattice)

    return new_atoms_, extend_lattice  # [[x1, y1, z1], [x2, y2, z2], etc.]


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


def calculate_twist_mos2(x_, y_, z_, angle_=0.0):
    # 括号条件表达式，对应于不同的层layer，其结果是正确的
    error_esp = 0.01
    # 不能保证十分准确，但是误差在1*10^-3之内
    # 修正了一个bug，与error值作比较时应取abs即“差的绝对值”，不然会导致预期之外的分类错误
    layer_s_1_up = (np.abs(z_ - z_layer_fold[0]) < error_esp)
    layer_s_1_down = (np.abs(z_ - z_layer_fold[1]) < error_esp)
    layer_mo_1 = (np.abs(z_ - z_layer_fold[2]) < error_esp)

    layer_s_2_up = (np.abs(z_ - z_layer_fold[3]) < error_esp)
    layer_s_2_down = (np.abs(z_ - z_layer_fold[4]) < error_esp)
    layer_mo_2 = (np.abs(z_ - z_layer_fold[5]) < error_esp)

    x_layer2, y_layer2, z_layer2 = x_[layer_s_1_up], y_[layer_s_1_up], z_[layer_s_1_up]
    x_layer3, y_layer3, z_layer3 = x_[layer_s_2_down], y_[layer_s_2_down], z_[layer_s_2_down]

    # with open('data/MoS2_1.data', 'w') as writer:
    #     writer.write('# MoS2 By Celeste\n\n')
    #     writer.write('%d atoms\n' % (len(z_)))
    #     writer.write('2 atom types\n\n')
    #     writer.write('%7.3f %7.3f xlo xhi\n' % (r2[0], r1[0]))
    #     writer.write('%7.3f %7.3f ylo yhi\n' % (r2[0], r2[1]))
    #     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, r3[2]))
    #     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (0.0, 0.0, 0.0))
    #     writer.write('  Masses\n\n')
    #     writer.write('1 32.07\n2 95.94\n\n')
    #     writer.write('Atoms\n\n')
    #
    #     index = 1
    #     for ix, iy, iz in zip(x_, y_, z_):
    #         if np.abs(iz - z_[2]) < error_esp:
    #             writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, ix, iy, iz))
    #         elif np.abs(iz - z_[5]) < error_esp:
    #             writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, ix, iy, iz))
    #         else:
    #             writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, ix, iy, iz))
    #         index += 1

    thetaAngle = np.deg2rad(angle_)
    x_layer3, y_layer3 = matrix_transformation(x_layer3, y_layer3, thetaAngle)

    layer_s_2_ = np.stack((x_layer2, y_layer2, z_layer2), axis=-1)
    layer_s_3_ = np.stack((x_layer3, y_layer3, z_layer3), axis=-1)

    return layer_s_2_, layer_s_3_


def calculate_layer_distance(first_layer, second_layer, **kwargs):
    print('func calculate_layer_distance()')
    # 最近的和最远的距离按打草纸上的图示
    # 在min-max区间内，用距离表示耦合关系
    minDistance = 4.30986  # = 9.59443 - 5.28457 不能再小了
    maxDistance = 4.68684
    mean_distance = (minDistance + maxDistance) * 0.5

    for first_atom in first_layer:
        distance_fs = distance.cdist(second_layer, [first_atom], 'euclidean')
        print(np.where(distance_fs < maxDistance))
        break
    distance_first = 0
    distance_second = 0

    with open('data/MoS2_2.data', 'w') as writer:
        writer.write('# MoS2 By Celeste\n\n')
        writer.write('%d atoms\n' % (2 * len(first_layer)))
        writer.write('2 atom types\n\n')
        writer.write('%7.3f %7.3f xlo xhi\n' % (r2[0], r1[0]))
        writer.write('%7.3f %7.3f ylo yhi\n' % (r2[0], r2[1]))
        writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, r3[2]))
        writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (0.0, 0.0, 0.0))
        writer.write('  Masses\n\n')
        writer.write('1 32.07\n2 32.07\n\n')
        writer.write('Atoms\n\n')

        index = 1
        for fl, sl in zip(first_layer, second_layer):
            writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, fl[0], fl[1], 5.28457))
            index += 1
            writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, sl[0], sl[1], 9.59443))
            index += 1
    # print(x_layer2, y_layer2)
    plt.figure(figsize=(6, 6))
    # plt.xticks([])
    # plt.yticks([])
    plt.plot(first_layer[:, 0], first_layer[:, 1], color='green', marker='o')
    plt.plot(second_layer[:, 0], second_layer[:, 1], color='red', marker='o')

    plt.show()
    print('finish')


def random_color():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


if __name__ == '__main__':
    # start here
    # 别忘了我们能通过公式推角度了。
    time1 = time.time()

    extend = 120
    file_path = os.path.join(os.getcwd(), 'data\\ratio_bonds_extend=%d.xlsx' % extend)
    if os.path.exists(file_path):
        os.remove(file_path)

    interesting_angle = [6.01, 7.34, 9.43, 10.42, 11.64, 13.17, 15.18, 16.43, 17.9,
                         21.79, 24.43, 26.01, 27.8, 29.41]

    a = 3.1903
    rt3 = 3 ** 0.5
    r1 = [rt3 / 2 * a, -1 / 2 * a, 0]
    r2 = [0, 1 * a, 0]
    r3 = [0, 0, 14.8790]
    mean_distance = 4.49835  # S1_up & S2_down mean distance

    new_atoms, ex_lattice = gen_mos2_lattice()

    x_mean = np.mean(new_atoms[:, 0])
    y_mean = np.mean(new_atoms[:, 1])
    xList = new_atoms[:, 0] - x_mean
    yList = new_atoms[:, 1] - y_mean
    zList = new_atoms[:, 2]
    z_layer_fold = zList[:6]
    x_drop, y_drop, z_drop = over_flow_drop(xList, yList, zList, x_mean)
    norm_xy_inequation = (normXY(x_drop, y_drop) > x_mean)
    # 判定方程都是一致的，保证drop对应原子一致
    norm_x = np.delete(x_drop, np.where(norm_xy_inequation))
    norm_y = np.delete(y_drop, np.where(norm_xy_inequation))
    norm_z = np.delete(z_drop, np.where(norm_xy_inequation))
    # 取模后保持一致，取中间圆形区域

    stack_layer_2, stack_layer_3 = calculate_twist_mos2(x_=norm_x, y_=norm_y, z_=norm_z, angle_=60.0)
    # plt.ion()
    # plt.figure(1)
    # calculate_layer_distance(stack_layer_2, stack_layer_3)

    result_list = []  # [[number, angle, area_ratio, bonds_ratio,], etc...]
    sum_step = 601
    # for agl in range(sum_step):
    #     # 先递增几个角度，每个角度的上下耦合数先算出来，算在区间距离内的原子个数，与当前层内原子数之比。
    #     coupling_ratio = 0.0
    #     stack_layer_2, stack_layer_3 = calculate_twist_mos2(x_=norm_x, y_=norm_y, z_=norm_z, angle_=agl / 10.0)
    #     print(".", end="")
    #     if agl % 50 == 0:
    #         print(' %d / %d' % (agl, sum_step - 1), end="\n")
    #     this_sum_bond = 0
    #     for first_atom in stack_layer_2:
    #         distance_fs = distance.cdist(stack_layer_3, [first_atom], 'euclidean')
    #         index = np.where(distance_fs < 4.68684 + 0.01)[0]
    #         # Sigma (bond_length from first to end atom)
    #         # 相对临界距离键长求和，表征上下耦合关系作fraction的分子， 分母是面积？(什么的面积？) mean_distance = 4.49835
    #         # 有的距离内是1 >> 1, 有的是 1 >> 2, 还不知道有没有1 >> 3
    #
    #         # print(distance_fs[np.where(distance_fs <= 4.68684+0.01)])
    #         this_sum_bond += sum(distance_fs[np.where(distance_fs < 4.68684 + 0.01)])  # bond_length
    #         # print(stack_layer_3[index], first_atom)
    #         # print('\t', stack_layer_3[index], '\t', first_atom, '\n')
    #         # rc = random_color()
    #         # plt.scatter(stack_layer_3[index][:, 0],
    #         #             stack_layer_3[index][:, 1], 30, marker='*', color=rc)
    #         # plt.scatter(np.array([first_atom])[:, 0], np.array([first_atom])[:, 1], 20, color=rc)
    #         # plt.draw()  # 注意此函数需要调用
    #
    #     denominator_area = np.pi * x_mean ** 2  # pi * R ** 2
    #     denominator_bonds = len(stack_layer_2) * mean_distance * 2  # 最多一个原子应该不会超过2个bond
    #     # print(this_sum_bond/denominator_area, this_sum_bond/denominator_bonds)
    #     result_list.append([agl, agl / 10.0, this_sum_bond / denominator_area, this_sum_bond / denominator_bonds])
    #     # break

    # xd = xlwt.Workbook()
    # sheet1 = xd.add_sheet('Sheet1')
    # title = ['number', 'angle', 'area_ratio', 'bonds_ratio']
    # row = 0
    # col = 0
    # for i in title:
    #     sheet1.write(row, col, i, style=xlwt.easyxf('font: bold on'))
    #     col += 1
    #
    # for i in result_list:
    #     row += 1
    #     col = 0
    #     for j in range(4):
    #         sheet1.write(row, col, i[j])
    #         col += 1
    #
    # xd.save(file_path)
    # print('finish')

    # plt.ioff()

    # plt.show()
    # print(norm_x, norm_y)
    # plt.figure(figsize=(6, 6))
    # # plt.xticks([])
    # # plt.yticks([])
    # plt.scatter(norm_x, norm_y)
    # plt.show()

    time2 = time.time()
    print('>> Finished, use time %d s ( %.2f min)' % (time2 - time1, (time2 - time1)/60))
