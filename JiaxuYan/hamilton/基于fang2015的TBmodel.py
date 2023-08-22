#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   基于fang2015的TBmodel.py    
@Time    :   2023/4/1 10:02
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    # t(1), t(2), t(3) = dt1, dt2, dt3
    # t(4), t(5) = (dt4, dt5), dt6
    # t(6)= (dt7, dt8, dt9)
    # 一共有 5d + 6p = 11轨道
    h[(3, 5)] = 2 * fir_t35 * np.cos(np.dot(wak, dt1)) + \
                sec_t35 * (np.exp(-1j * np.dot(wak, dt2)) + np.exp(-1j * np.dot(wak, dt3))) + \
                thi_t35 * (np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
    h[(6, 8)] = 2 * fir_t68 * np.cos(np.dot(wak, dt1)) + \
                sec_t68 * (np.exp(-1j * np.dot(wak, dt2)) + np.exp(-1j * np.dot(wak, dt3))) + \
                thi_t68 * (np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
    h[(9, 11)] = 2 * fir_t9B * np.cos(np.dot(wak, dt1)) + \
                 sec_t9B * (np.exp(-1j * np.dot(wak, dt2)) + np.exp(-1j * np.dot(wak, dt3))) + \
                 thi_t9B * (np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
    h[(1, 2)] = -2j * fir_t12 * np.sin(np.dot(wak, dt1)) + \
                    sec_t12 * (np.exp(-1j * np.dot(wak, dt2)) - np.exp(-1j * np.dot(wak, dt3))) + \
                    thi_t12 * (-np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
    if i < n * 20:  # 判断i的大小确定k的取值
        kx = i / (n * 20) * K1[0]
        ky = 0
    elif i < n * 30:
        ky = (i - n * 20) / (n * 10) * (Middle[1])
        kx = K1[0] - ky*np.tan(pi/6)
    else:
        [kx, ky] = Middle * (1-(i - n * 30) / (n * 17))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import functools
import pybinding as pb


# table MoS2 has checked
# table MoS2 has checked
def Table7(MX2='MoS2'):
    name_ = MX2
    eps_dic = {
        # ->       eps1,   eps2,    eps3,    eps4,    eps5,    eps6,   eps7,   eps8,    eps9,    epsA,   epsB,
        'MoS2': [1.0688, 1.0688, -0.7755, -1.2902, -1.2902, -0.1380, 0.0874, 0.0874, -2.8949, -1.9065, -1.9065],
        'MoSe2': [0.7819, 0.7819, -0.6567, -1.1726, -1.1726, -0.2297, 0.0149, 0.0149, -2.9015, -1.7806, 1.7806],
        'WS2': [1.3754, 1.3754, -1.1278, -1.5534, -1.5534, -0.0393, 0.1984, 0.1984, -3.3706, -2.3461, -2.3461],
        'WSe2': [1.0349, 1.0349, -0.9573, -1.3937, -1.3937, -0.1667, 0.0984, 0.0984, -3.3642, -2.1820, -2.1820],
    }
    # onsite refer sphere (n=1) Tij (i==j)
    onsite_dic = {
        # ->      t(1)11, t(1)22,  t(1)33, t(1)44,  t(1)55,  t(1)66, t(1)77,  t(1)88,  t(1)99, t(1)AA, t(1)BB (十六进制)
        'MoS2': [-0.2069, 0.0323, -0.1739, 0.8651, -0.1872, -0.2979, 0.2747, -0.5581, -0.1916, 0.9122, 0.0059],
        'MoSe2': [-0.1460, 0.0177, -0.2112, 0.9638, -0.1724, -0.2636, 0.2505, -0.4734, -0.2166, 0.9911, -0.0036],
        'WS2': [-0.2011, 0.0263, -0.1749, 0.8726, -0.2187, -0.3716, 0.3537, -0.6892, -0.2112, 0.9673, 0.0143],
        'WSe2': [-0.1395, 0.0129, -0.2171, 0.9763, -0.1985, -0.3330, 0.3190, -0.5837, -0.2399, 1.0470, 0.0029],
    }
    # offsite refer sphere (n=1) Tij (i!=j)
    offsite_dic = {
        # ->      t(1)35, t(1)68, t(1)9B,  t(1)12,  t(1)34,  t(1)45,  t(1)67,  t(1)78, t(1)9A, t(1)AB (十六进制)
        'MoS2': [-0.0679, 0.4096, 0.0075, -0.2562, -0.0995, -0.0705, -0.1145, -0.2487, 0.1063, -0.0385],
        'MoSe2': [-0.0735, 0.3520, 0.0047, -0.1912, -0.0755, -0.0680, -0.0960, -0.2012, 0.1216, -0.0394],
        'WS2': [-0.0818, 0.4896, -0.0315, -0.3106, -0.1105, -0.0989, -0.1467, -0.3030, 0.1645, -0.1018],
        'WSe2': [-0.0912, 0.4233, -0.0377, -0.2321, -0.0797, -0.0920, -0.1250, -0.2456, 0.1857, -0.1027],
    }
    # offsite refer sphere (n=5,6) Tij (i!=j)
    other_dic = {
        # ->      t(5)41, t(5)32,  t(5)52, t(5)96,  t(5)B6,  t(5)A7, t(5)98,  t(5)B8,  t(6)96,  t(6)B6, t(6)98, t(6)B8
        'MoS2': [-0.7883, -1.3790, 2.1584, -0.8836, -0.9402, 1.4114, -0.9535, 0.6517, -0.0686, -0.1498, -0.2205,
                 -0.2451],
        'MoSe2': [-0.6946, -1.3258, 1.9415, -0.7720, -0.8738, 1.2677, -0.8578, 0.5545, -0.0691, -0.1553, -0.2227,
                  -0.2154],
        'WS2': [-0.8855, -1.4376, 2.3121, -1.0130, -0.9878, 1.5629, -0.9491, 0.6718, -0.0659, -0.1533, -0.2618,
                -0.2736],
        'WSe2': [-0.7744, -1.4014, 2.0858, -0.8998, -0.9044, 1.4030, -0.8548, 0.5711, -0.0676, -0.1608, -0.2618,
                 -0.2424],
    }
    # 返回值：[0:11]是eps能量，[11:22]是eps能量，[22:32]是eps能量，[32:]是eps能量，
    return eps_dic[name_] + onsite_dic[name_] + offsite_dic[name_] + other_dic[name_]


def plot_rec(*args, **kwargs):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    ax.arrow(0, 0, a1[0], a1[1], length_includes_head=False, head_width=0.05, fc='b', ec='k')
    ax.arrow(0, 0, a2[0], a2[1], length_includes_head=False, head_width=0.05, fc='b', ec='k')
    ax.arrow(0, 0, b1[0], b1[1], length_includes_head=False, head_width=0.05, fc='r', ec='red')
    ax.arrow(0, 0, b2[0], b2[1], length_includes_head=False, head_width=0.05, fc='r', ec='red')
    ax.plot([0, Middle[0], K1[0], 0], [0, Middle[1], K1[1], 0])
    ax.scatter([0, Middle[0], K1[0], 0], [0, Middle[1], K1[1], 0])
    ax.set_xlim(-2, 4)
    ax.set_ylim(-1, 4)
    ax.grid()
    plt.show()


def hexij(i_, j_):
    if i_ >= 10:
        i_ = hex(i_).split('x')[1].upper()
    else:
        i_ = str(i_)
    if j_ >= 10:
        j_ = hex(j_).split('x')[1].upper()
    else:
        j_ = str(j_)
    return i_, j_


def HamHam(wak):
    h = np.zeros((11, 11), dtype=complex)
    # Equation (4) check  # diagonal elements
    for ii in range(11):
        h[ii, ii] = eps_1[ii] + 2 * onsite_1[ii] * np.cos(np.dot(wak, dt1)) + \
                    2 * onsite_2[ii] * (np.cos(np.dot(wak, dt2)) + np.cos(np.dot(wak, dt3)))

    # Equation (5) check # 这里是可以转为字典进行遍历的, 还方便了添加共轭项
    for (ii, jj) in [(3, 5), (6, 8), (9, 11)]:
        firij = dij['fir_t%s%s' % hexij(ii, jj)]
        secij = dij['sec_t%s%s' % hexij(ii, jj)]
        thiij = dij['thi_t%s%s' % hexij(ii, jj)]
        ii -= 1
        jj -= 1
        h[(ii, jj)] = 2 * firij * np.cos(np.dot(wak, dt1)) + \
                      secij * (np.exp(-1j * np.dot(wak, dt2)) + np.exp(-1j * np.dot(wak, dt3))) + \
                      thiij * (np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
        h[(jj, ii)] = np.conj(h[(ii, jj)]).T

    # Equation (6) check 如果没有改进的话，这里要写 nX3行
    for (ii, jj) in [(1, 2), (3, 4), (4, 5), (6, 7), (7, 8), (9, 10), (10, 11)]:
        firij = dij['fir_t%s%s' % hexij(ii, jj)]
        secij = dij['sec_t%s%s' % hexij(ii, jj)]
        thiij = dij['thi_t%s%s' % hexij(ii, jj)]
        ii -= 1
        jj -= 1
        h[(ii, jj)] = -2j * firij * np.sin(np.dot(wak, dt1)) + \
                      secij * (np.exp(-1j * np.dot(wak, dt2)) - np.exp(-1j * np.dot(wak, dt3))) + \
                      thiij * (-np.exp(1j * np.dot(wak, dt2)) + np.exp(1j * np.dot(wak, dt3)))
        h[(jj, ii)] = np.conj(h[(ii, jj)]).T
    # Equation (7) check
    for (ii, jj) in [(3, 1), (5, 1), (4, 2), (10, 6), (9, 7), (11, 7), (10, 8)]:
        fouij = dij['fou_t%s%s' % hexij(ii, jj)]
        ii -= 1
        jj -= 1
        h[(ii, jj)] = fouij * (np.exp(1j * np.dot(wak, dt4)) - np.exp(1j * np.dot(wak, dt6)))
        h[(jj, ii)] = np.conj(h[(ii, jj)]).T
    # Equation (8) check
    for (ii, jj) in [(4, 1), (3, 2), (5, 2), (9, 6), (11, 6), (10, 7), (9, 8), (11, 8)]:
        fouij = dij['fou_t%s%s' % hexij(ii, jj)]
        fivij = dij['fiv_t%s%s' % hexij(ii, jj)]
        ii -= 1
        jj -= 1
        h[(ii, jj)] = fouij*(np.exp(1j * np.dot(wak, dt4)) + np.exp(1j * np.dot(wak, dt6))) + \
                      fivij*np.exp(1j*np.dot(wak, dt5))
        h[(jj, ii)] = np.conj(h[(ii, jj)]).T

    return h


if __name__ == '__main__':
    # start here

    a = 3.160  # 3.18
    c = 12.29  # distance of layers
    dxx = 3.13  # distance of orbital X-X
    dxm = 2.41  # distance of orbital X-M
    # constant checked
    rt3 = 3 ** 0.5
    pi = np.pi
    a1 = a * np.array([1, 0])
    a2 = a * np.array([-1 / 2, rt3 / 2])
    b1 = 2 * pi / a * np.array([1, 1 / rt3])
    b2 = 4 * pi / a / rt3 * np.array([0, 1])
    Gamma = np.array([0, 0])
    Middle = 1 / 2 * b1
    K1 = 1 / 3 * (2 * b1 - b2)
    K2 = -1 / 3 * (2 * b1 - b2)
    # plot_rec()

    dt1 = a1
    dt2 = a1 + a2
    dt3 = a2
    dt4 = -(2 * a1 + a2) / 3
    dt5 = (a1 + 2 * a2) / 3
    dt6 = (a1 - a2) / 3
    dt7 = -2 * (a1 + a2) / 3
    dt8 = 2 * (2 * a1 + a2) / 3
    dt9 = 2 * (a2 - a1) / 3

    mx = 'MoS2'
    T_MoS2 = Table7(MX2=mx)
    dij = {}
    # eps1, eps2, eps3, eps4, eps5, eps6, eps7, eps8, eps9, eps10, eps11 = T_MoS2[:11]
    fir_t11, fir_t22, fir_t33, fir_t44, fir_t55, fir_t66, fir_t77, fir_t88, fir_t99, fir_tAA, fir_tBB = T_MoS2[11:22]
    fir_t35, fir_t68, fir_t9B, fir_t12, fir_t34, fir_t45, fir_t67, fir_t78, fir_t9A, fir_tAB = T_MoS2[22:32]
    fiv_t41, fiv_t32, fiv_t52, fiv_t96, fiv_tB6, fiv_tA7, fiv_t98, fiv_tB8, six_t96, six_tB6, six_t98, six_tB8 = \
        T_MoS2[32:]
    dij['fir_t35'], dij['fir_t68'], dij['fir_t9B'], dij['fir_t12'] = fir_t35, fir_t68, fir_t9B, fir_t12
    dij['fir_t34'], dij['fir_t45'], dij['fir_t67'], dij['fir_t78'] = fir_t34, fir_t45, fir_t67, fir_t78
    dij['fir_t9A'], dij['fir_tAB'] = fir_t9A, fir_tAB

    dij['fiv_t41'], dij['fiv_t32'], dij['fiv_t52'], dij['fiv_t96'] = fiv_t41, fiv_t32, fiv_t52, fiv_t96
    dij['fiv_tB6'], dij['fiv_tA7'], dij['fiv_t98'], dij['fiv_tB8'] = fiv_tB6, fiv_tA7, fiv_t98, fiv_tB8
    dij['six_t96'], dij['six_tB6'], dij['six_t98'], dij['six_tB8'] = six_t96, six_tB6, six_t98, six_tB8

    # second diagonal onsite checked
    sec_t11 = 1 / 4 * fir_t11 + 3 / 4 * fir_t22
    sec_t22 = 3 / 4 * fir_t11 + 1 / 4 * fir_t22
    sec_t33 = fir_t33
    sec_t44 = 1 / 4 * fir_t44 + 3 / 4 * fir_t55
    sec_t55 = 3 / 4 * fir_t44 + 1 / 4 * fir_t55
    sec_t66 = fir_t66
    sec_t77 = 1 / 4 * fir_t77 + 3 / 4 * fir_t88
    sec_t88 = 3 / 4 * fir_t77 + 1 / 4 * fir_t88
    sec_t99 = fir_t99
    sec_tAA = 1 / 4 * fir_tAA + 3 / 4 * fir_tBB
    sec_tBB = 3 / 4 * fir_tAA + 1 / 4 * fir_tBB
    # second and third offsite alpha beta gamma checked
    # gamma beta
    sec_t35 = rt3 / 2 * fir_t34 - 1 / 2 * fir_t35
    thi_t35 = -rt3 / 2 * fir_t34 - 1 / 2 * fir_t35
    sec_t68 = rt3 / 2 * fir_t67 - 1 / 2 * fir_t68
    thi_t68 = -rt3 / 2 * fir_t67 - 1 / 2 * fir_t68
    sec_t9B = rt3 / 2 * fir_t9A - 1 / 2 * fir_t9B
    thi_t9B = -rt3 / 2 * fir_t9A - 1 / 2 * fir_t9B
    dij['sec_t35'], dij['thi_t35'], dij['sec_t68'], dij['thi_t68'], dij['sec_t9B'], dij['thi_t9B'] = \
        sec_t35, thi_t35, sec_t68, thi_t68, sec_t9B, thi_t9B
    # alpha beta checked
    sec_t12 = rt3 / 4 * (fir_t11 - fir_t22) - fir_t12
    thi_t12 = -rt3 / 4 * (fir_t11 - fir_t22) - fir_t12
    sec_t45 = rt3 / 4 * (fir_t44 - fir_t55) - fir_t45
    thi_t45 = -rt3 / 4 * (fir_t44 - fir_t55) - fir_t45
    sec_t78 = rt3 / 4 * (fir_t77 - fir_t88) - fir_t78
    thi_t78 = -rt3 / 4 * (fir_t77 - fir_t88) - fir_t78
    sec_tAB = rt3 / 4 * (fir_tAA - fir_tBB) - fir_tAB
    thi_tAB = -rt3 / 4 * (fir_tAA - fir_tBB) - fir_tAB
    dij['sec_t12'], dij['thi_t12'], dij['sec_t45'], dij['thi_t45'] = sec_t12, thi_t12, sec_t45, thi_t45
    dij['sec_t78'], dij['thi_t78'], dij['sec_tAB'], dij['thi_tAB'] = sec_t78, thi_t78, sec_tAB, thi_tAB
    # gamma alpha checked
    sec_t34 = 1 / 2 * fir_t34 + rt3 / 2 * fir_t35
    thi_t34 = 1 / 2 * fir_t34 - rt3 / 2 * fir_t35
    sec_t67 = 1 / 2 * fir_t67 + rt3 / 2 * fir_t68
    thi_t67 = 1 / 2 * fir_t67 - rt3 / 2 * fir_t68
    sec_t9A = 1 / 2 * fir_t9A + rt3 / 2 * fir_t9B
    thi_t9A = 1 / 2 * fir_t9A - rt3 / 2 * fir_t9B
    dij['sec_t34'], dij['thi_t34'], dij['sec_t67'], dij['thi_t67'], dij['sec_t9A'], dij['thi_t9A'] = \
        sec_t34, thi_t34, sec_t67, thi_t67, sec_t9A, thi_t9A
    # fourth layer alpha beta gamma
    fou_t41 = 1 / 4 * fiv_t41 + 3 / 4 * fiv_t52
    fou_t52 = 3 / 4 * fiv_t41 + 1 / 4 * fiv_t52
    fou_tA7 = 1 / 4 * fiv_tA7 + 3 / 4 * fiv_tB8
    fou_tB8 = 3 / 4 * fiv_tA7 + 1 / 4 * fiv_tB8
    fou_t42 = -rt3 / 4 * fiv_t41 + rt3 / 4 * fiv_t52
    fou_t51 = fou_t42
    fou_tA8 = -rt3 / 4 * fiv_tA7 + rt3 / 4 * fiv_tB8
    fou_tB7 = fou_tA8

    fou_t31 = -rt3 / 2 * fiv_t32
    fou_t97 = -rt3 / 2 * fiv_t98
    fou_t32 = -1 / 2 * fiv_t32
    fou_t98 = -1 / 2 * fiv_t98

    fou_t96 = fiv_t96
    fou_tA6 = -rt3 / 2 * fiv_tB6
    fou_tB6 = -1 / 2 * fiv_tB6
    dij['fou_t41'], dij['fou_t52'], dij['fou_tA7'], dij['fou_tB8'] = fou_t41, fou_t52, fou_tA7, fou_tB8
    dij['fou_t42'], dij['fou_t51'], dij['fou_tA8'], dij['fou_tB7'] = fou_t42, fou_t51, fou_tA8, fou_tB7
    dij['fou_t31'], dij['fou_t97'], dij['fou_t32'], dij['fou_t98'] = fou_t31, fou_t97, fou_t32, fou_t98
    dij['fou_t96'], dij['fou_tA6'], dij['fou_tB6'] = fou_t96, fou_tA6, fou_tB6

    eps_1 = T_MoS2[:11]
    onsite_1 = T_MoS2[11:22]
    offsite_1 = T_MoS2[22:32]
    othersite_1 = T_MoS2[32:]
    onsite_2 = [sec_t11, sec_t22, sec_t33, sec_t44, sec_t55, sec_t66, sec_t77, sec_t88, sec_t99, sec_tAA, sec_tBB]

    hamilton = functools.partial(HamHam)
    gamma = [0, 0]
    path = pb.make_path(gamma, Middle, K1, gamma, step=0.001)
    result = np.zeros([len(path), 11])  # 解的矩阵
    n = 0
    for kxy in path:
        k = np.array(kxy)  # 得到k值，带入D矩阵
        w, t = np.linalg.eig(hamilton(k))
        w = list(w)
        w.sort()
        result[n, :] = np.real(w)  # 将本征值进行保存
        n += 1
    xk = [0, rt3, rt3+1, rt3+3]
    kk = np.linspace(0, 4.7, num=len(path))  # 作为x轴，使其和本征值矩阵每一列的y的值个数相同
    plt.figure(figsize=(4, 5))
    plt.plot(kk, result, c="r")
    plt.xlim(0, rt3+3)
    plt.xticks(xk, ["Γ", "M", "K", "Γ"])
    plt.ylabel("Energy(eV)", fontsize=14)
    plt.axvline(rt3, color='gray', linestyle='--')
    plt.axvline(rt3 + 1, color='gray', linestyle='--')
    # plt.title('%s' % mx)
    plt.tick_params(labelsize=18)
    plt.show()
    # k = [1/2, rt3/2]
    # print(np.dot(k, dt2))
    print("finished")


