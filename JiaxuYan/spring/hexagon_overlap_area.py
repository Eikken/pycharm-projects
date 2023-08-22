#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   hexagon_overlap_area.py
@Time    :   2022/11/24 15:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   hexagon_overlap_area.py
将数据归一化到[a,b]区间范围的方法：
（1）首先找到样本数据Y的最小值Min及最大值Max
（2）计算系数为：k=（b-a)/(Max-Min)
（3）得到归一化到[a,b]区间的数据：norY=a+k(Y-Min)
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
    函数图像拟合 http://t.zoukankan.com/heaiping-p-9068401.html
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from scipy.optimize import curve_fit
from ase.calculators.h2morse import ome
from prettytable import PrettyTable


def heron_formula(*args, **kwargs):
    theta_ = args[0]
    acc_ = 2.46 / 3 ** 0.5
    vfcb = np.pi / 6 - theta_ / 2
    bc = acc_ / 2  # a acc？
    fc = bc / np.cos(vfcb)  # c
    vfce = 2 * np.pi / 3
    vcef = theta_
    vefc = np.pi / 3 - vcef
    # 三角形efc三个角和fc边知道了，求另外两个边。
    ef = fc * np.sin(vfce) / np.sin(vfce + vefc)  # a
    ec = fc * np.sin(vefc) / np.sin(vfce + vefc)  # b
    p = np.sum([ef, ec, fc]) / 2
    return (p * (p - ef) * (p - ec) * (p - fc)) ** 0.5


def cal_omega(beta_):
    pi = np.pi  # pi
    c_ = 3 * 10 ** 10  # light speed m/s
    Fc_ = beta_  # area A^2
    param_ = 1
    mass_mX = [7.56, 7.54, 30.2, 44.7, 53.2, 46.7, 59.2]  # 0:G 1:hBN 2:MoS2 3:MoSe2 4:MoTe2 5:WS2 6:WSe2
    mass_ = [m * 10 ** -7 for m in mass_mX]  # kg/m^2
    sin_pi_4 = np.sin(pi / 4)
    w = param_ * (Fc_ / mass_[2]) ** 0.5 * sin_pi_4 / (pi * c_)
    # 存在这种可能 : y = a*x^2 + b*x +c
    # 当前beta是未知量，通过theta求解beta，然后做 omega(theta) 求解
    return w * 10 ** 7  # 10^8量级


def epf(x_, a_, b_):  # exp 指数拟合
    return 1 / (a_ * np.exp(b_ / x_))


if __name__ == '__main__':
    # start here
    table = PrettyTable(['angle', 'lambda', 'shadow', 'Fc', 'ratio', 'omega'])  # Fc is hexagon overlap area
    plt.figure(figsize=(4, 6))

    x = np.linspace(0, 4, 400)  # 36>>30
    y = 0.375 * x ** 2 - 3 * x + 36
    Xb = []
    freq = -1.5 * x + 36
    comp = np.stack((x, freq, y), axis=-1)
    plt.plot(comp[:, 0], comp[:, 1])
    plt.plot(comp[:, 0], comp[:, 2])

    # # 归一化计算
    #
    for i in range(1, 401):
        angle = i / 100.0
        theta = np.deg2rad(angle)
        a = 2.46
        acc = a / 3 ** 0.5
        lam = acc / (2 * np.sin(theta / 2))
        shadow_area = heron_formula(theta)
        Xb.append(acc / (2 * np.sin(np.pi * 0.1 / 180 + theta / 2)))
        Fc = 3 * 3 ** 0.5 * lam ** 2 / 2 - 6 * shadow_area  # 正六边形面积
        ratio = round(shadow_area * 100 / Fc, 6)  # 这个ratio和angle是线性单调变化的关系
        omega = cal_omega(Fc)
        row = [round(angle, 4), round(lam, 2), round(shadow_area, 2), round(Fc, 2),
               ratio, omega]
        # omega 除以二次函数才能逼近图像
        table.add_row(row)
    la = np.array(table.rows)

    #
    # plt.plot(la[:, 0], norm_omega, linestyle="--")
    # plt.show()
    # print(table)

    # z1 = np.polyfit(x, la[:, 5]/y, 4)
    # p1 = np.poly1d(z1)
    # print(p1)
    # yv = p1(x)  # 多项式拟合
    yv = la[1:, 5] / freq[1:]
    popt, pcov = curve_fit(epf, x[1:], yv, maxfev=500000)
    a = popt[0]  # popt里面是拟合系数，读者可以自己help其用法
    b = popt[1]
    print("func = %.3f * exp(%.3f/x)" % (a, b))
    # func = 4.133 * exp(-0.036/x) - 0.1
    yv = epf(x, a, b) - 0.1
    # plt.scatter(la[:, 0], la[:, 5] / freq, 1)
    # plt.plot(x, y, linestyle="--")
    # plt.plot(la[3:, 0], la[3:, 5]/yv[3:], linestyle="-.")
    la5 = la[:, 5] / yv
    k = (36 - 30) / (np.max(la5) - np.min(la5))  # 系数
    norm_omega = k * (la5 - np.min(la5)) + 30
    plt.plot(la[:, 0], norm_omega, linestyle=":")
    plt.show()
    # for i, j, k in zip(y, la[:, 5], la[:, 0]):
    #     print(' | ', k, ' | ', j/i, ' | ')
