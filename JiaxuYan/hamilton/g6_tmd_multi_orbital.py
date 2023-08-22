#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   g6_tmd_multi_orbital.py    
@Time    :   2022/9/13 10:44  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import re
import time
import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
import pybinding.repository.group6_tmd as g6tmd
from scipy.spatial import cKDTree


def SK_params(*args):
    r'''
    Parameters
    ----------
    l m n : float
        Vector of hopping orientation a1 -a2 a1-a2.
    V_x : float
        Slater-Koster parameters.
    H_dd : numpy array
        dd orbital hamiltonian matrix.

    # zF = z^2
    # xFyF = x^2 - y^2
    # xy = xy
    这是来自 t0 t1 的上一步！错误示范！
    '''

    V_sigma = -2.8  # eV  dd orbital
    V_pi = 0.1  # eV
    V_delta = 0.1  # eV

    [l, m, n] = args[0]  # vector

    # ->
    zF_zF = (n ** 2 - 1 / 2 * (l ** 2 + m ** 2)) ** 2 * V_sigma \
            + 3 * n ** 2 * (l ** 2 + m ** 2) * V_pi \
            + 3 / 4 * (l ** 2 + m ** 2) ** 2 * V_delta

    xFyF_zF = 3 ** 0.5 / 2 * (l ** 2 - m ** 2) * (n ** 2 - (l ** 2 + m ** 2) / 2) * V_sigma \
              + 3 ** 0.5 * n ** 2 * (m ** 2 - l ** 2) * V_pi \
              + 3 ** 2 / 4 * (1 + n ** 2) * (l ** 2 - m ** 2) * V_delta

    xFyF_xFyF = 3 / 4 * (l ** 2 - m ** 2) ** 2 * V_sigma \
                + (l ** 2 + m ** 2 - (l ** 2 - m ** 2) ** 2) * V_pi \
                + (n ** 2 + (l ** 2 - m ** 2) ** 2 / 4) * V_delta

    xy_zF = 3 ** 0.5 * l * m * (n ** 2 - (l ** 2 + m ** 2) / 2) * V_sigma \
            - 2 * 3 ** 0.5 * l * m * n ** 0.5 * V_pi \
            + 3 ** 0.5 / 2 * l * m * (1 + n ** 2) * V_delta

    xy_xFyF = 3 / 2 * l * m * (l ** 2 - m ** 2) * V_sigma \
              + 2 * l * m * (m ** 2 - l ** 2) * V_pi \
              + 1 / 2 * l * m * (l ** 2 - m ** 2) * V_delta

    xy_xy = 3 * l ** 2 * m ** 2 * V_sigma \
            + (l ** 2 + m ** 2 - 4 * l ** 2 * m ** 2) * V_pi \
            + (n ** 2 + l ** 2 * m ** 2) * V_delta

    # 共轭添加矩阵
    H_dd = np.array([[   zF_zF,   xFyF_zF,   xy_zF],
                     [-xFyF_zF, xFyF_xFyF, xy_xFyF],
                     [  -xy_zF,  -xy_xFyF, xy_xy]])
    return H_dd


def twist_layers(theta):
    theta = theta / 180 * math.pi  # from degrees to radians
    twistMatrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    @pb.site_position_modifier
    def rotate(x, y, z):
        """Rotate layer 2 by the given angle `theta`"""
        layer2 = (z < 0)
        x0 = x[layer2]
        y0 = y[layer2]
        x[layer2] = x0 * math.cos(theta) - y0 * math.sin(theta)
        y[layer2] = y0 * math.cos(theta) + x0 * math.sin(theta)
        return x, y, z

    @pb.hopping_generator('interlayer', energy=0.1)  # eV
    def interlayer_generator(x, y, z):
        """Generate hoppings for site pairs which have distance `d_min < d < d_max`"""
        positions = np.stack([x, y, z], axis=1)
        layer1 = (z == 0)
        layer2 = (z != 0)

        d_min = c0 * 0.96
        d_max = c0 * 1.1
        kdtree1 = cKDTree(positions[layer1])
        kdtree2 = cKDTree(positions[layer2])
        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

        idx = coo.data > d_min
        abs_idx1 = np.flatnonzero(layer1)
        abs_idx2 = np.flatnonzero(layer2)
        row, col = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]
        return row, col  # lists of site indices to connect

    @pb.hopping_energy_modifier
    def interlayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
        """Set the value of the newly generated hoppings as a function of distance"""
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)[:, 0, 0]
        interlayer = (hop_id == 'interlayer')  # energy是3*3
        energy[interlayer] = 0.4 * c0 / d[interlayer]
        return energy

    return rotate, interlayer_generator, interlayer_hopping_value


def TMD_dd_lat(*args, **kwargs):
    name_ = kwargs['name']

    params = g6tmd._default_3band_params.copy()
    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = params[name_]
    rt3 = math.sqrt(3)  # convenient constant

    lat = pb.Lattice(a1=[a, 0], a2=[1 / 2 * a, rt3 / 2 * a])

    metal_name, chalcogenide_name = re.findall("[A-Z][a-z]*", name_)  # Mo S

    lat.add_one_sublattice(metal_name+'1', [0, 0, 0], [eps1, eps2, eps2])  # layer 1
    lat.add_one_sublattice(metal_name+'2', [0, 0, -c0], [eps1, eps2, eps2])  # layer 2

    lmn = [[1, 0, 0],
           [-1 / 2, -3 ** 0.5 / 2, 0],
           [-1 / 2, -3 ** 0.5 / 2, 0]]

    H1 = SK_params(lmn[0])  # a1
    H2 = SK_params(lmn[1])  # -a2
    H3 = SK_params(lmn[2])  # a1-a2

    m1, m2 = metal_name+'1', metal_name+'2'
    lat.add_hoppings(([1, 0], m1, m1, H1),
                     ([0, -1], m1, m1, H2),
                     ([1, -1], m1, m1, H3),

                     ([1, 0], m2, m2, H1),
                     ([0, -1], m2, m2, H2),
                     ([1, -1], m2, m2, H3)
                     )

    lat.min_neighbors = 3

    return lat


def initial_g6tmd():
    model = pb.Model(g6tmd.monolayer_3band("MoS2"), pb.translational_symmetry())
    solver = pb.solver.lapack(model)

    k_points = model.lattice.brillouin_zone()
    gamma = [0, 0]
    k = k_points[0]
    m = (k_points[0] + k_points[1]) / 2

    plt.figure(figsize=(5, 7))

    plt.subplot(211, title="MoS2 3-band model band structure")
    bands = solver.calc_bands(gamma, k, m, gamma)
    bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    plt.subplot(212, title="Band structure path in reciprocal space")
    model.lattice.plot_brillouin_zone(decorate=False)
    bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    # initial_g6tmd()

    # creat_bigger_lat(ex=2, name="MoS2")
    c0 = 0.650
    # this_lattice = TMD_dd_lat(name='MoS2')
    #
    this_model = pb.Model(g6tmd.monolayer_3band("MoS2"),
                          pb.translational_symmetry(),
                          # pb.circle(radius=1.0),
                          )
    solver = pb.solver.lapack(this_model)

    k_points = this_model.lattice.brillouin_zone()
    Gamma = [0, 0]
    K = k_points[0]
    M = (k_points[0] + k_points[1]) / 2
    #
    # plt.figure(figsize=(5, 8), dpi=100)
    # # plt.subplot(211, title="xy plane")
    # this_model.plot()
    # this_model.lattice.plot_vectors(position=[0, 0])
    #
    # # # plt.subplot(212, title="xz plane")
    # # # this_model.plot(axes='xz')
    # #
    # plt.savefig(r'png/MoS2-2.png', dpi=200)
    # plt.show()
    # plt.clf()

    plt.figure(figsize=(5, 7), dpi=100)
    plt.subplot(211, title="MoS2 3-band model band structure")
    bands = solver.calc_bands(Gamma, K, M, Gamma)
    bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    plt.subplot(212, title="Band structure path in reciprocal space")
    this_model.lattice.plot_brillouin_zone(decorate=False)
    bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    plt.show()
    plt.clf()

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
