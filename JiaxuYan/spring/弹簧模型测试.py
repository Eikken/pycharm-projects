#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   弹簧模型测试.py    
@Time    :   2022/11/7 11:19  
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


def solve_omega(*args, **kwargs):
    # 每个角度下：beta123都相同； mass123与lamda(theta)有关；只有一个omega值
    # mass & beta 对应成比例
    # beta/mass is constant == Omega
    pi_ = np.pi
    c_ = 3 * np.power(10, 8)  # 光速 light speed
    omg = 36.0
    sin_pi2 = np.sin(pi_/2)

    lam_ = lam_value(a_cc_=acc, theta_=angle)

    beta = 1.0
    beta1 = (4.0+2 * 3 ** 0.5) * beta
    beta6 = 3.0 * beta
    beta4 = 2 * 3 ** 0.5 * beta
    beta3 = 1.0 * beta

    mass = lam_
    mass1 = (4.0+2 * 3 ** 0.5) * lam_
    mass6 = 3.0 * mass
    mass4 = 2 * 3 ** 0.5 * mass
    mass3 = 1.0 * mass
    omg = sin_pi2 * (beta1/mass1)**0.5 / (pi_ * c_)
    arr = []
    for i in range(1, 14):
        agl_ = i / 10.0
        lam_ = lam_value(a_cc_=acc, theta_=agl_)
        mass1 = (4.0 + 2 * 3 ** 0.5) * lam_
        omg = sin_pi2 * (beta1 / mass1) ** 0.5 / (pi_)

        arr.append([agl_, omg])
        print("angle :\t%.2f\t omg :\t%.2f\t lamda :\t%.2f\t" % (agl_, omg, lam_))
    arr = np.array(arr)
    # plt.figure(figsize=(6, 8))
    plt.plot(arr[:, 0], arr[:, 1], marker='o', color='k', linewidth=1)
    plt.show()


def lam_value(a_cc_, theta_):
    if theta_ == 0:
        begin_color = '\033[1;31m'
        end_color = '\033[0m'
        print(begin_color, "RuntimeWarning: theta is zero, divide by zero encountered in double_scalars", end_color)
        return 0.0
    theta_ = np.deg2rad(theta_)
    return a_cc_ / (2 * (np.sin(theta_ / 2.0)))


if __name__ == '__main__':
    # start here
    acc = 2.4595 / 3 ** 0.5
    rt2 = 2 ** 0.5

    # 通过lam_value 获取对应角度下的lamda值。
    angle = 1.0
    lam = lam_value(a_cc_=acc, theta_=angle)
    solve_omega()
    # arr = []
    # for i in range(1, 14):
    #     angle = i / 10.0
    #     lam = lam_value(a_cc_=acc, theta_=angle)
    #     arr.append([angle, lam])
    #     print("angle :\t%.2f\t  lamda value :\t%.2f\t" % (angle, lam))
    # arr = np.array(arr)
    # # plt.figure(figsize=(6, 8))
    # plt.plot(arr[:, 0], arr[:, 1], marker='o', color='k', linewidth=1)
    # plt.show()