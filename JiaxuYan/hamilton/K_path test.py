#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   K_path test.py
@Time    :   2022/8/16 14:46
@E-mail  :   iamwxyoung@qq.com
@Tips    :   arpack 求解器 指定k值
             lapack 无指定k值，即对于mode path进行set wave vector求解本征值。
'''

import time

import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
from pybinding.repository import graphene

from JiaxuYan.hamilton import my_twist_constants

from JiaxuYan.pbtest.genPosDataStructure import gen_lattice


def this_sort(item):
    return int(item[1:])


def your_func_here(*args, **kwargs):
    pass


def kpath_func(*args, **kwargs):

    Gamma_ = [0, 0]
    M_ = [0, 2 * np.pi / (3 * lat_acc)]
    K2_ = [2 * np.pi / (3 * 3 ** 0.5 * lat_acc), 2 * np.pi / (3 * lat_acc)]
    K1_ = [-4 * np.pi / (3 * 3 ** 0.5 * lat_acc), 0]

    this_path_model = pb.Model(
        lattice,
        pb.translational_symmetry()
    )
    this_path = pb.results.make_path(Gamma_, K2_, M_, Gamma_, step=0.01)

    # print(this_path)
    this_path_model.plot()
    this_path_model.lattice.plot_vectors(position=[0, 0])
    plt.title('path model')
    plt.show()
    plt.clf()

    solver = pb.solver.lapack(this_path_model)

    path_bands = []  # 重写path band方法

    for kp in this_path:
        solver.set_wave_vector(kp)
        path_bands.append(solver.eigenvalues)
    result = pb.results.Bands(this_path, path_bands)

    plt.figure(dpi=100)
    result.plot_kpath()
    # result.plot(lw=0.5)

    hh = 5

    # plt.ylim(-hh, hh)
    plt.title('this path bands')
    plt.show()
    plt.clf()

    # kx_lim = pi / graphene.a
    # kx_path = np.linspace(-kx_lim, kx_lim, 200)  # 4.1pi
    # ky_outer = 0
    # ky_inner = 2 * pi / (3 * graphene.a_cc)  # 4.7pi
    # outer_bands = []
    # for kx in kx_path:
    #     solver.set_wave_vector([kx, ky_outer])
    #     outer_bands.append(solver.eigenvalues)
    #
    # inner_bands = []
    # for kx in kx_path:
    #     solver.set_wave_vector([kx, ky_inner])
    #     inner_bands.append(solver.eigenvalues)
    #
    # for bands in [outer_bands, inner_bands]:
    #     result = pb.results.Bands(kx_path, bands)
    #     result.plot()
    #     # result.plot_kpath()
    # plt.title('path bands')
    # plt.show()
    # plt.clf()


def lapack_func(*args, **kwargs):
    this_la_model = pb.Model(lattice,
                             pb.translational_symmetry(),
                             # pb.rectangle(x=18),
                             )
    this_la_model.plot()
    this_la_model.lattice.plot_vectors(position=[-8, 10])  # nm
    plt.title('this_la_model')
    plt.show()
    plt.clf()

    solver = pb.solver.lapack(this_la_model)
    eigenvalues = solver.calc_eigenvalues()
    eigenvalues.plot_heatmap(show_indices=True)
    plt.title('this_la_model eigenvalues')
    pb.pltutils.colorbar()
    plt.show()
    plt.clf()


def arpack_func(*args, **kwargs):
    this_ar_model = pb.Model(lattice,
                             pb.circle(radius=15),
                             # pb.translational_symmetry(),
                             )
    # this_ar_model.plot()
    # this_ar_model.lattice.plot_vectors(position=[0, 0])  # nm
    # plt.title('this_ar_model')
    # plt.show()
    # plt.clf()

    solver = pb.solver.arpack(this_ar_model, k=30)

    at_qu = [1.4, 1.6]
    eigenvalues = solver.calc_eigenvalues(map_probability_at=at_qu)

    # 只有这一个map_probability_at可以plot密度的点
    plt.figure(figsize=(7, 5))
    eigenvalues.plot_heatmap(show_indices=False)
    plt.title('this_ar [%.2f, %.2f] eigenvalues' % (at_qu[0], at_qu[1]))
    pb.pltutils.colorbar()
    # return solver
    plt.show()
    plt.clf()

    prob = 6
    probability_map = solver.calc_probability(prob)
    probability_map.plot(site_radius=(0.2, 0.3))
    plt.title('this_ar [%.2f, %.2f] prod=%d' % (at_qu[0], at_qu[1], prob))
    plt.show()
    plt.clf()

    erg = -0.85
    brod = 0.03

    ldos_map = solver.calc_spatial_ldos(energy=erg, broadening=brod)  # [eV]
    ldos_map.plot(site_radius=(0.2, 0.3))
    plt.title('this_ar_ldos energy=%.2f' % erg)
    plt.show()
    plt.clf()

    dos = solver.calc_dos(energies=np.linspace(-2, 2, 888), broadening=brod)  # [eV]
    dos.plot()
    plt.title('this_ar_dos broadening=%.2f' % brod)
    plt.show()
    plt.clf()


def ldos_plot_func(*args, **kwargs):
    this_dos_model = pb.Model(lattice,
                              pb.circle(radius=15),
                              )
    # this_dos_model.plot()
    # this_dos_model.lattice.plot_vectors(position=[-5, 7])  # nm
    # plt.title('this_dos_model')
    # plt.show()

    solver = pb.solver.arpack(this_dos_model, k=20)  # k 决定选择最底下的特征值
    energy_list = [i * 0.05 for i in range(35)]
    ldos_map = solver.calc_spatial_ldos(energy=0, broadening=0.05)  # [eV]
    ldos_map.plot()

    plt.show()
    plt.clf()

    # solver.set_wave_vector(k=0)

    # for energy in energy_list:
    #     ldos = solver.calc_spatial_ldos(energy=energy, broadening=0.05)  # eV
    #     ldos.plot(site_radius=(0.2, 0.3))
    #     pb.pltutils.colorbar(label="LDOS")
    #     plt.title('%.2f eV k1' % energy)
    #     plt.savefig('png/ldos/%.2f eV LDOS plot.png' % energy, dpi=100)
    #     # plt.show()
    #     plt.clf()
    #     solver.clear()
    #     print(energy_list.index(energy), ' / 35')


def bands_plot_func(*args, **kwargs):
    label_list = kwargs['label_path']
    this_band_model = pb.Model(lattice,
                               pb.translational_symmetry(),
                               )

    # this_band_model.plot()
    # this_band_model.lattice.plot_vectors(position=[-5, 7])  # nm
    # plt.title('this_band_model')
    # plt.show()

    solver = pb.solver.arpack(this_band_model, k=14)  # k 决定选择最底下的特征值

    bands = solver.calc_bands(args[0], args[1], args[2], args[3], step=0.001)
    bands.plot(point_labels=label_list, lw=0.75)
    plt.show()
    plt.clf()

    this_band_model.lattice.plot_brillouin_zone(decorate=False)
    bands.plot_kpath(point_labels=label_list, lw=0.75)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    lat_acc = my_twist_constants.a_lat_21_7 / 3 ** 0.5

    lattice, all_hopping_acc = gen_lattice()
    shape = pb.circle(radius=15)
    model = pb.Model(lattice,
                     pb.translational_symmetry(),
                     # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
                     # shape,
                     )
    broadening = 0.03

    Gamma = [0, 0]
    M = [0, 2 * np.pi / (3 * lat_acc)]
    K2 = [2 * np.pi / (3 * 3 ** 0.5 * lat_acc), 2 * np.pi / (3 * lat_acc)]
    K1 = [-4 * np.pi / (3 * 3 ** 0.5 * lat_acc), 0]
    # path 1 : Gamma >> K >> M >> Gamma
    # path 2 : K1 >> Gamma >> M >> K2
    path_list_1 = [r'$\Gamma$', 'K', 'M', r'$\Gamma$']
    path_list_2 = ['K1', r'$\Gamma$', 'M', 'K2']

    # bands_plot_func(Gamma, K2, M, Gamma, label_path=path_list_1)
    # bands_plot_func(K1, Gamma, M, K2, label_path=path_list_2)
    # ldos_plot_func()
    # lapack_func()
    # arpack_func()
    # kpath_func()
    this_la_model = pb.Model(lattice,
                             pb.translational_symmetry(),
                             # pb.rectangle(x=18),
                             )
    this_la_model.plot()
    this_la_model.lattice.plot_vectors(position=[-8, 10])  # nm
    plt.title('this_la_model')
    plt.show()
    plt.clf()

    solver = pb.solver.lapack(this_la_model)
    k_points = model.lattice.brillouin_zone()
    gamma = [0, 0]
    k = k_points[0]
    m = (k_points[0] + k_points[1]) / 2

    bands = solver.calc_bands(gamma, k, m, gamma, step=0.05)
    bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    plt.title('this_la_model eigenvalues')
    plt.ylim(-5, 5)
    plt.show()
    plt.clf()

    plt.title("Band structure path in reciprocal space")
    model.lattice.plot_brillouin_zone(decorate=False)
    bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])

    plt.show()
    plt.clf()

    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
