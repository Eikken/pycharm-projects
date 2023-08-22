#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   multi_orbital.py    
@Time    :   2022/8/24 11:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time


# import math
import matplotlib.pyplot as plt
import pybinding as pb
# import pandas as pd
import numpy as np
from pybinding.repository import group6_tmd, graphene

from JiaxuYan.hamilton import my_twist_constants
from JiaxuYan.pbtest.genPosDataStructure import gen_lattice


def your_func_here(*args, **kwargs):
    pass


@pb.onsite_energy_modifier
def potential(energy, x):
    """Linear onsite potential as a function of x for a 2-orbital model"""
    return energy + np.eye(2) * x


@pb.onsite_energy_modifier
def potential(energy, x, sub_id):
    """Applies different functions to different sublattices"""
    if sub_id == "A":
        return energy + x  # we know sublattice A is single-orbital
    elif sub_id == "D":
        energy[x > 0] += sub_id.eye * x  # the notation can be mixed with numpy indexing
        return energy                    # apply only to sites where x > 0
    elif sub_id == "B":
        sigma_y = np.array([[0, -1j],
                            [1j,  0]])
        return energy + sigma_y * 1.3 - np.eye(2) * 0.6  # add multiple 2x2 matrices
    else:
        return energy  # leave the other sublattices unchanged


def onsite_hopping_plot(*args, **kwargs):
    lattice_ = kwargs['lat']
    lat = pb.Lattice([1, 0], [0, 1])
    lat.add_sublattices(
        ("A", [0.0, 0.0], 0.5),  # single-orbital: scalar
        ("B", [0.0, 0.2], [[1.5, 2j],  # two-orbital: 2x2 Hermitian matrix
                           [-2j, 1.5]]),
        ("C", [0.3, 0.1], np.zeros(2)),  # two-orbital: zero onsite term
        ("D", [0.1, 0.0], [[4, 0, 0],  # three-orbital: only diagonal
                           [0, 5, 0],
                           [0, 0, 6]]),
        ("E", [0.2, 0.2], [4, 5, 6])  # three-orbital: only diagonal, terse notation
    )

    lat.add_hoppings(
        ([0, 0], "B", "E", [[2j, 1, 2], [1j, 3, 4]]),

        ([0, 1], "A", "A", 1.2),  # scalar
        ([0, 1], "B", "B", [[1, 2],  # 2x2
                            [3, 4]]),
        ([0, 0], "B", "C", [[2j, 0],  # 2x2
                            [1j, 0]]),
        ([0, 0], "A", "D", [[1, 2, 3]]),  # 1x3
        ([0, 1], "D", "A", [[7],  # 3x1
                            [8],
                            [9]]),
        ([0, 0], "B", "D", [[1j, 0, 0],  # 2x3
                            [2, 0, 3j]])
    )

    this_model = pb.Model(
        lat,
        pb.primitive(2, 2)
    )

    return this_model


def TMDs_multi_orbital_plot(*args, **kwargs):

    a, eps1, eps2, t0, t1, t2, t11, t12, t22 = [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057]
    rt3 = np.sqrt(3)  # convenient constant

    lat = pb.Lattice(a1=[a, 0], a2=[1 / 2 * a, rt3 / 2 * a])

    h1 = [[t0, -t1, t2],
          [t1, t11, -t12],
          [t2, t12, t22]]

    h2 = [[t0, 1 / 2 * t1 + rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [-1 / 2 * t1 + rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t11 - t22) - t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t11 - t22) + t12, 3 / 4 * t11 + 1 / 4 * t22]]

    h3 = [[t0, -1 / 2 * t1 - rt3 / 2 * t2, rt3 / 2 * t1 - 1 / 2 * t2],
          [1 / 2 * t1 - rt3 / 2 * t2, 1 / 4 * t11 + 3 / 4 * t22, rt3 / 4 * (t22 - t11) + t12],
          [-rt3 / 2 * t1 - 1 / 2 * t2, rt3 / 4 * (t22 - t11) - t12, 3 / 4 * t11 + 1 / 4 * t22]]

    m = 'C1'
    lat.add_hoppings(([1, 0], m, m, h1),
                     ([0, -1], m, m, h2),
                     ([1, -1], m, m, h3))

    return lat


def graphene_multi_orbital_plot(*args, **kwargs):

    shape = pb.rectangle(4, 5)

    a_lat = 2.46

    lat = pb.Lattice(a1=[a_lat, 0, 0], a2=[a_lat / 2, a_lat / 2 * np.sqrt(3), 0])

    cpos = [[1/3, 3**0.5/3], [1/3, 2*3**0.5/3], [2/3, 3**0.5/3], [2/3, 2*3**0.5/3], [1/2, 3**0.5/2]]

    for i, j in zip('ABCDE', range(5)):
        lat.add_sublattices(
            (i, [k * a_lat for k in cpos[j]])  # [i*6.51172 for i in cpos])
        )
        print((i, [k * a_lat for k in cpos[j]]))

    t = -2.8
    lat.add_hoppings(
        ([0, 0], 'A', 'E', t),
        ([0, 0], 'B', 'E', t),
        ([0, 0], 'C', 'E', t),
        ([0, 0], 'D', 'E', t),
    )

    this_model = pb.Model(
        lat,  # .with_offset(position=[graphene.a_cc/2, 0]),
        shape
    )

    this_model.plot()
    this_model.lattice.plot_vectors(position=[0, 0])
    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    graphene_multi_orbital_plot()

    # ------------------------------------------------------------------
    # lat_acc = my_twist_constants.a_lat_21_7 / 3 ** 0.5
    #
    # lattice, all_hopping_acc = gen_lattice()
    #
    # # model = onsite_hopping_plot(lat=lattice)
    #
    # model = pb.Model(group6_tmd.monolayer_3band("MoS2"),
    #                  pb.regular_polygon(6, 20))
    #
    # kpm = pb.kpm(model)
    #
    # # plt.figure(figsize=(3, 3), dpi=200)
    # # model.plot()
    # # plt.show()
    # # plt.clf()
    #
    # energy = np.linspace(-1, 3.8, 500)
    # broadening = 0.05
    # position = [0, 0]
    #
    # plt.figure(figsize=(7.5, 3))
    #
    # plt.subplot(121, title="Reduced -- sum of all orbitals")
    # ldos = kpm.calc_ldos(energy, broadening, position)
    # ldos.plot(color='C1')
    #
    # plt.subplot(122, title="Individual orbitals")
    # ldos = kpm.calc_ldos(energy, broadening, position, reduce=False)
    # ldos.plot()
    #
    # plt.show()
    # plt.clf()
    #
    # dos = kpm.calc_dos(energy, broadening, num_random=20)
    # dos.plot()
    #
    # plt.show()
    # plt.clf()
    #################################################################
    # print(model.system.num_sites)
    # # 20  # <-- 5 sites per unit cell and 2x2 cells: 5*2*2 == 20
    # print(model.hamiltonian.shape)
    # # (44, 44)  # <-- 11 (1+2+2+3+3) orbitals per unit cell and 2x2 cells: 11*2*2 = 44

    # grid = plt.GridSpec(1, 3, hspace=0)
    # plt.figure(figsize=(7.5, 3), dpi=200)
    #
    # for g, p in zip(grid, ['xy', 'zy', 'zx']):
    #     plt.subplot(g, title=p)
    #     model.plot(axes=p)
    #     # model.lattice.plot(axes=p)
    # plt.show()
    # plt.clf()
    #
    # sys_idx = model.system.find_nearest(position=[0, 0], sublattice="D")
    #
    # print(sys_idx)
    # # 15 <-- Points to a site on sublattice D which is closest to the target position.
    # print(model.system.x[sys_idx])
    # print(model.system.y[sys_idx])
    #
    # ham_idx = model.system.to_hamiltonian_indices(sys_idx)
    # print(ham_idx)  #     Size 3 because the selected site is on the 3-orbital sublattice D.
    # ham = model.hamiltonian.todense()
    # print(ham[np.ix_(ham_idx, ham_idx)])  # Returns the onsite hopping term of sublattice D.

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))