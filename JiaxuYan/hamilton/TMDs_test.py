#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   TMDs_test.py    
@Time    :   2022/8/23 15:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   Monolayer of a group 6 TMD using the nearest-neighbor 3-band model
             The available options are:
                MoS2, WS2, MoSe2, WSe2, MoTe2, WTe2.
             override_params()
                Replace or add new material parameters.
                The dictionary entries must be in the format
                "name": [a, eps1, eps2, t0, t1, t2, t11, t12, t22].
'''
import itertools
import time
import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
import pybinding.repository.group6_tmd as g6tmd


def your_func_here(*args, **kwargs):
    pass


def g6tmds_band_plot(*args, **kwargs):
    grid = plt.GridSpec(2, 3, hspace=0.4)
    plt.figure(figsize=(12, 7.5), dpi=100)

    for square, name in zip(grid, ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]):
        model = pb.Model(g6tmd.monolayer_3band(name), pb.translational_symmetry())
        solver = pb.solver.lapack(model)

        k_points = model.lattice.brillouin_zone()
        gamma = [0, 0]
        k = k_points[0]
        m = (k_points[0] + k_points[1]) / 2

        plt.subplot(square, title=name)
        bands = solver.calc_bands(gamma, k, m, gamma)
        bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"], lw=1.5)

    plt.show()
    plt.clf()


def mos2_band_plot(*args, **kwargs):
    #
    # g6tmd.monolayer_3band("MoS2").plot()
    # plt.show()
    # plt.clf()
    mos2_a = g6tmd._default_3band_params["MoS2"][0]

    shape1 = pb.circle(radius=1.2)
    shape2 = pb.regular_polygon(num_sides=6, radius=1.2),
    this_model = pb.Model(g6tmd.monolayer_3band("MoS2"),
                          pb.translational_symmetry(),
                          )

    broadening = 0.1

    # this_model.plot()
    # this_model.lattice.plot_vectors(position=[0, 0])
    #
    # plt.show()
    # plt.clf()
    #
    # this_model.lattice.plot_brillouin_zone()
    #
    # plt.show()
    # plt.clf()
    # grid = plt.GridSpec(1, 3, hspace=0)
    # plt.figure(figsize=(7.5, 3), dpi=150)
    #
    # for g, p in zip(grid, ['xy', 'yz', 'xz']):
    #     plt.subplot(g, title=p)
    #     this_model.plot(axes=p)
    #
    # plt.show()
    # plt.clf()
    #
    # k_points = this_model.lattice.brillouin_zone()

    # print([i/np.pi for i in k_points])
    # solver = pb.solver.lapack(this_model)

    # k_ = 30
    # solver = pb.solver.arpack(this_model, k=k_, )
    #
    # k_points = this_model.lattice.brillouin_zone()
    # gamma = [0, 0]
    # k = k_points[1]  # convenient get K points reciprocal vector
    # m = (k_points[0] + k_points[1]) / 2
    #
    # this_path = pb.results.make_path(gamma, k, m, gamma, step=0.1)
    #
    # path_eigenvalues = []
    #
    # for tp in this_path:
    #     solver.set_wave_vector(tp)
    #     path_eigenvalues.append(solver.eigenvalues)
    #
    # result = pb.results.Bands(this_path, path_eigenvalues)

    # return this_path, np.array(path_eigenvalues)
    # print(len(path_eigenvalues[0]))

    # plt.figure(figsize=(7, 3), dpi=100)
    #
    # plt.subplot(121, title="MoS2 band structure")
    # for h in range(len(path_eigenvalues)):
    #     plt.plot(this_path, np.array(path_eigenvalues)[, h])
    #     print(np.array(path_eigenvalues)[, :])
    #
    # # result.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"],)
    #
    # plt.subplot(122, title="Band path")
    # result.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    #
    # hh = 5
    #
    # # plt.ylim(-hh, hh)
    # plt.title('override path bands')
    # plt.show()
    # plt.clf()

    # plt.figure(figsize=(7, 3), dpi=100)

    # plt.subplot(121, title="MoS2 band structure")
    # bands = solver.calc_bands(gamma, k, m, gamma)
    # bands.plot(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    #
    # plt.subplot(122, title="Band path")
    # this_model.lattice.plot_brillouin_zone(decorate=False)
    # bands.plot_kpath(point_labels=[r"$\Gamma$", "K", "M", r"$\Gamma$"])
    # plt.show()
    # plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    # mos2_band_plot()
    # g6tmds_band_plot()
    #
    # plt.scatter(np.arange(len(t_path)), t_eigenvalues[:, 0])
    #
    # plt.show()
    rt3 = math.sqrt(3)
    lat = pb.Lattice(a1=[1, 0], a2=[1 / 2, rt3 / 2])
    v1 = lat.reciprocal_vectors()
    print([v for v in v1])

    # a = np.array([[1, 2], [3, 4]])
    # ainv = np.linalg.inv(a)
    # print(ainv, '\n', ainv.T)
    #
    # print([ns for ns in itertools.product([-1, 0, 1], repeat=2)])

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
