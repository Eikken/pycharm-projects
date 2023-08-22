#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   new_lambda.py    
@Time    :   2022/12/4 10:20  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   用来绘制上下层特殊角重合点

'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from JiaxuYan.spring.different_radius_ratio import *


def nl_cal_euclidean_s1_s2_(s_1, s_2):
    acc_ = 0.01
    dis1 = distance.cdist(s_1, s_2, 'euclidean').min(axis=1)
    dis2 = distance.cdist(s_1, s_2, 'euclidean').min(axis=0)
    # 先取三分之一acc吧
    index_s1 = np.where(dis1 < acc_)
    index_s2 = np.where(dis2 < acc_)

    return index_s1, index_s2


if __name__ == '__main__':
    # start here
    time1 = time.time()

    Super = 160
    sub_value = 2
    a_cc = 1.42 / sub_value
    norm_x, norm_y, norm_z = genGraphene(Super=Super)
    angle_list = [4.41, 5.09, 6.01, 7.34, 8.61, 9.43, 10.42, 11.64, 13.17, 15.18, 16.43, 17.90, 19.65, 21.79, 24.43,
                  26.01, 27.8, 18.73, 19.03, 19.27, 19.65, 20.32, 23.48, 24.02, 25.04, 25.46, 28.78, 29.41]
    layer_init = np.stack([norm_x, norm_y, norm_z], axis=1)
    fig_size = 15
    la = []
    print(sorted(angle_list))
    for i in [4.41]:
        angle = i
        print(angle)
        theta = np.deg2rad(angle)
        twist_x, twist_y = matrix_transformation(norm_x, norm_y, theta=theta)
        layer_twist = np.stack([twist_x, twist_y, norm_z], axis=1)
        index_S1, index_S2 = nl_cal_euclidean_s1_s2_(layer_init[:, :2], layer_twist[:, :2])
        tmp_init = layer_init[index_S1]
        tmp_twist = layer_twist[index_S2]
        # index_ = np.where(distance.cdist(tmp_twist, tmp_twist) 0)
        dis_arr = distance.cdist(tmp_twist, tmp_twist)
        print(dis_arr[:, 0])
        plt.figure(figsize=(fig_size, fig_size))
        plt.title("angle = %.2f° " % angle)
        plt.scatter(layer_init[:, 0], layer_init[:, 1], 5, color='g', alpha=0.5)
        plt.scatter(layer_twist[:, 0], layer_twist[:, 1], 5, color='b', alpha=0.5)
        plt.scatter(tmp_init[:, 0], tmp_init[:, 1], 30, color='r')
        plt.scatter(tmp_twist[:, 0], tmp_twist[:, 1], 30, color='r')
        # plt.savefig('png/特殊角重合点图样/angle_%.2f.png' % i)
        plt.show()
        # time.sleep(1)
        plt.clf()
    # for i in range(401, 3011):
    #     print(".", end="")
    #     if i % 50 == 0:
    #         print('. (echo %d / %d)' % (i, 3010), end="\n")
    #     angle = i / 100.0
    #     # 100*100 21.79° template has 1914 pair of atoms. 4.41° template has 84 pair of atoms.
    #     # 80*80 21.79° template has 1218 pair of atoms. 4.41° template has 54 pair of atoms.
    #     theta = np.deg2rad(angle)
    #     twist_x, twist_y = matrix_transformation(norm_x, norm_y, theta=theta)
    #     layer_twist = np.stack([twist_x, twist_y, norm_z], axis=1)
    #     index_S1, index_S2 = nl_cal_euclidean_s1_s2_(layer_init[:, :2], layer_twist[:, :2])
    #     tmp_init = layer_init[index_S1]
    #     tmp_twist = layer_twist[index_S2]
    #     if len(tmp_init) >= 54:
    #         la.append([angle, len(tmp_init) / 1914.0, len(tmp_init)])
    #
    # la = np.array(la)
    # plt.title('%dx%d' % (Super, Super))
    # plt.plot(la[:, 0], la[:, 1], marker='o')
    # plt.show()
    # pd.DataFrame(la).to_excel('data/%dx%d.xls' % (Super, Super))
    # if len(tmp_init) >= 6:
    #     print(angle, len(tmp_init))
    #     plt.figure(figsize=(fig_size, fig_size))
    #     plt.title("angle = %.2f° " % angle)
    #     plt.scatter(layer_init[:, 0], layer_init[:, 1], 10, color='g', alpha=0.5)
    #     plt.scatter(layer_twist[:, 0], layer_twist[:, 1], 10, color='b', alpha=0.5)
    #     plt.scatter(tmp_init[:, 0], tmp_init[:, 1], 30, color='r')
    #     plt.scatter(tmp_twist[:, 0], tmp_twist[:, 1], 30, color='r')
    #     # plt.savefig('png/angle_%d.png' % i)
    #     plt.show()
    #     time.sleep(7)
    #     plt.close()
    time2 = time.time()
    print('>> Finished, use time %d min' % ((time2 - time1) / 60))
