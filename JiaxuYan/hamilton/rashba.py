#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   rashba.py    
@Time    :   2022/8/25 14:43  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import re
import time

import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt


def your_func_here(*args, **kwargs):
    pass


"""Calculate the band structure of graphene with Rashba spin-orbit coupling"""


def monolayer_graphene_soc():
    """Return the lattice specification for monolayer graphene with Rashba SOC,
       see http://doi.org/10.1103/PhysRevB.95.165415 for reference"""
    from pybinding.constants import pauli
    from pybinding.repository.graphene import a_cc, a, t

    onsite = 0.05  # [eV] onsite energy
    rashba = 0.1  # [eV] strength of Rashba SOC
    rashba_so = 1j * 2 / 3 * rashba

    # create a lattice with 2 primitive vectors
    a1 = np.array([a / 2 * sqrt(3), a / 2])
    a2 = np.array([a / 2 * sqrt(3), -a / 2])
    lat = pb.Lattice(
        a1=a1, a2=a2
    )

    pos_a = np.array([-a_cc / 2, 0])
    pos_b = np.array([+a_cc / 2, 0])

    lat.add_sublattices(
        ('A', pos_a, [[onsite, 0], [0, onsite]]),
        ('B', pos_b, [[-onsite, 0], [0, -onsite]]))

    # nearest neighbor vectors
    d1 = (pos_b - pos_a) / a_cc  # [ 0,  0]
    d2 = (pos_b - pos_a - a1) / a_cc  # [-1,  0]
    d3 = (pos_b - pos_a - a2) / a_cc  # [ 0, -1]

    nn_hopp = np.array([[t, 0], [0, t]])  # nn hopping, same spin
    t1 = nn_hopp + rashba_so * (pauli.x * d1[1] - pauli.y * d1[0])  # cross([sx , sy], [dx, dy])
    t2 = nn_hopp + rashba_so * (pauli.x * d2[1] - pauli.y * d2[0])
    t3 = nn_hopp + rashba_so * (pauli.x * d3[1] - pauli.y * d3[0])
    # name and position
    lat.add_hoppings(
        ([0, 0], 'A', 'B', t1),
        ([-1, 0], 'A', 'B', t2),
        ([0, -1], 'A', 'B', t3)
    )

    return lat


def geng6tmds_lat(name):

    _default_3band_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
        # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22
        "MoS2": [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338, 0.057],
        "WS2": [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061],
        "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290, 0.130],
        "WSe2": [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329, 0.034],
        "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239, 0.252],
        "WTe2": [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270, 0.190],
    }
    params = _default_3band_params.copy()

    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params[name]
    rt3 = np.sqrt(3)  # convenient constant

    lat = pb.Lattice(a1=[a, 0], a2=[1/2 * a, rt3/2 * a])

    metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", name)
    lat.add_one_sublattice(metal_name, [0, 0], [eps1, eps2, eps2])

    h1 = [[ t0, -t1,   t2],
          [ t1, t11, -t12],
          [ t2, t12,  t22]]

    h2 = [[                    t0,     1/2 * t1 + rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [-1/2 * t1 + rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t11 - t22) - t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t11 - t22) + t12,     3/4 * t11 + 1/4 * t22]]

    h3 = [[                    t0,    -1/2 * t1 - rt3/2 * t2,     rt3/2 * t1 - 1/2 * t2],
          [ 1/2 * t1 - rt3/2 * t2,     1/4 * t11 + 3/4 * t22, rt3/4 * (t22 - t11) + t12],
          [-rt3/2 * t1 - 1/2 * t2, rt3/4 * (t22 - t11) - t12,     3/4 * t11 + 1/4 * t22]]

    m = metal_name
    lat.add_hoppings(([1, 0], m, m, h1),
                     ([0, -1], m, m, h2),
                     ([1, -1], m, m, h3))
    # lat.add_hoppings(([1,  0], m, m, h1),
    #                  ([0, -1], m, m, np.transpose(h2)),
    #                  ([1, -1], m, m, np.transpose(h3)))
    return lat


def k_path_tmds():

    m_cc = 0.3190

    a1 = [1, 0]
    a2 = [1 / 2, 3 ** 0.5 / 2]

    Gamma_ = [0, 0]
    K1_ = [2 * pi / 3 / m_cc, 2 * pi / 3 ** 0.5 / m_cc]
    M1_ = [0, 2 * np.pi / (3 ** 0.5 * m_cc)]

    K2_ = [4 * pi / 3 / m_cc, 0]
    M2_ = [pi/m_cc, pi/3**0.5/m_cc]

    K3_ = [-2 * pi / 3 / m_cc, -2 * pi / 3 ** 0.5 / m_cc]
    M3_ = [0, -2 * pi / 3 ** 0.5 / m_cc]

    this_path_1 = pb.results.make_path(Gamma_, K1_, M1_, Gamma_, step=0.01)
    this_path_2 = pb.results.make_path(Gamma_, K2_, M2_, Gamma_, step=0.01)
    this_path_3 = pb.results.make_path(Gamma_, K3_, M3_, Gamma_, step=0.01)

    this_model = pb.Model(geng6tmds_lat("MoS2"),
                          # pb.circle(0.65),
                          pb.translational_symmetry()
                          )

    this_model.plot()
    this_model.lattice.plot_vectors(position=[0, 0])
    plt.title('this model')
    plt.show()
    plt.clf()

    solver = pb.solver.lapack(this_model)

    path_bands = []
    #
    # for kp in this_path_1:
    #     solver.set_wave_vector(kp)
    #     path_bands.append(solver.eigenvalues)
    # result = pb.results.Bands(this_path_1, path_bands)
    #
    # this_model.lattice.plot_brillouin_zone()
    # result.plot_kpath()
    # plt.title('brillouin path')
    # plt.show()
    # plt.clf()
    #
    # result.plot()
    # plt.title('path band result')
    # plt.show()
    # plt.clf()

    # for kp in this_path_2:
    #     solver.set_wave_vector(kp)
    #     path_bands.append(solver.eigenvalues)
    # result = pb.results.Bands(this_path_2, path_bands)
    #
    # this_model.lattice.plot_brillouin_zone()
    # result.plot_kpath()
    # plt.title('brillouin path')
    # plt.show()
    # plt.clf()
    #
    # result.plot()
    # plt.title('path band result')
    # plt.show()
    # plt.clf()

    for kp in this_path_3:
        solver.set_wave_vector(kp)
        path_bands.append(solver.eigenvalues)
    result = pb.results.Bands(this_path_3, path_bands)

    this_model.lattice.plot_brillouin_zone()

    result.plot_kpath()
    plt.title('brillouin path')
    plt.show()
    plt.clf()

    result.plot()
    plt.title('path band result')
    plt.show()
    plt.clf()

    for i in this_model.lattice.reciprocal_vectors():
        print(i/np.pi)
    return result


def plot_sin_cos():

    plt.figure(figsize=(5, 4), dpi=200)

    plt.subplot(211, title="sin-y")
    # bands = solver.calc_bands(gamma, k, m, gamma)
    # bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')  # 将右边 边沿线颜色设置为空 其实就相当于抹掉这条边
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # 设置中心的为（0，0）的坐标轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    ax.set_xticks(np.arange(-2*pi, 2*pi+1, pi))
    ax.set_yticks([])
    xx = np.linspace(-2*pi, 2*pi, 200)
    siny = np.sin(xx)
    cosy = np.cos(xx)

    ax.plot(xx, siny)

    plt.subplot(212, title="cos-y")
    # model.lattice.plot_brillouin_zone(decorate=False)
    # bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    ax = plt.gca()  # get current axis 获得坐标轴对象
    ax.spines['right'].set_color('none')  # 将右边 边沿线颜色设置为空 其实就相当于抹掉这条边
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # 设置中心的为（0，0）的坐标轴
    ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
    ax.spines['left'].set_position(('data', 0))
    ax.set_xticks(np.arange(-2 * pi, 2 * pi + 1, pi))
    ax.set_yticks([])
    ax.plot(xx, cosy)

    plt.show()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    res = k_path_tmds()

    # j, q = np.arange(len(res.energy[:, 0])), res.energy[:, 0]
    #
    # plt.figure(dpi=200)
    # plt.scatter(j, q)
    # plt.show()

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
