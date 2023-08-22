#!/usr/bin python
# -*- encoding: utf-8 -*-
"""
@Author  :   Celeste Young
@File    :   pbTMDstest.py
@Time    :   2022/7/26 16:01
@E-mail  :   iamwxyoung@qq.com
@Tips    :
"""

import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np


def twist_bilayer_graphene():

    a_ = 6.51172
    r2 = [-1/2 * a_, -3**0.5/2 * a_]
    r1 = [1*a_, 0]
    # r1 = [a_ / 2, a_ / 2 * math.sqrt(3)]
    # r2 = [a_ / 2, -a_ / 2 * math.sqrt(3)]
    # r2 = [-3.25586, -5.63931]
    # r1 = [6.51172, 0.00000]
    lat = pb.Lattice(a1=r1, a2=r2)
    dataSet = pd.read_csv(r'data\1.xyz').values
    for i in range(1, len(dataSet)):
        # cpos = list(map(float, [dataSet[i][0].split(' ')[-5], dataSet[i][0].split(' ')[-3], dataSet[i][0].split('
        # ')[-1]]))
        cpos = [float(j) for j in dataSet[i][0].split(' ')[1:] if j != ''][:3]  # 笛卡尔坐标

        lat.add_sublattices(
            ('C'+str(i), [j*a_ for j in cpos])# [i*6.51172 for i in cpos])
        )

    lat.register_hopping_energies({
        'gamma0': -2.8,  # [eV] intralayer
        'gamma1': -0.4,  # [eV] interlayer
    })
    lat.add_hoppings(
        # layer 1
        ([0, 0], 'C1', 'C4', 'gamma0'),
        ([0, 0], 'C2', 'C5', 'gamma0'),
        ([0, 0], 'C2', 'C3', 'gamma0'),
        ([0, 0], 'C3', 'C6', 'gamma0'),
        ([0, 0], 'C4', 'C5', 'gamma0'),
        ([0, 0], 'C4', 'C7', 'gamma0'),
        ([0, 0], 'C5', 'C8', 'gamma0'),
        ([0, 0], 'C6', 'C9', 'gamma0'),
        ([0, 0], 'C7', 'C10', 'gamma0'),
        ([0, 0], 'C8', 'C9', 'gamma0'),
        ([0, 0], 'C8', 'C11', 'gamma0'),
        ([0, 0], 'C10', 'C13', 'gamma0'),
        ([0, 0], 'C10', 'C11', 'gamma0'),
        ([0, 0], 'C11', 'C14', 'gamma0'),
        ([0, 0], 'C12', 'C13', 'gamma0'),

        # layer 2
        ([0, 0], 'C15', 'C18', 'gamma0'),
        ([0, 0], 'C16', 'C19', 'gamma0'),
        ([0, 0], 'C17', 'C18', 'gamma0'),
        ([0, 0], 'C17', 'C20', 'gamma0'),
        ([0, 0], 'C18', 'C21', 'gamma0'),
        ([0, 0], 'C19', 'C20', 'gamma0'),
        ([0, 0], 'C20', 'C23', 'gamma0'),
        ([0, 0], 'C21', 'C22', 'gamma0'),
        ([0, 0], 'C21', 'C24', 'gamma0'),
        ([0, 0], 'C22', 'C25', 'gamma0'),
        ([0, 0], 'C23', 'C24', 'gamma0'),
        ([0, 0], 'C23', 'C26', 'gamma0'),
        ([0, 0], 'C24', 'C27', 'gamma0'),
        ([0, 0], 'C25', 'C28', 'gamma0'),
        ([0, 0], 'C27', 'C28', 'gamma0'),

        # layer 1
        ([0, a_], 'C1', 'C12', 'gamma0'),
        ([0, a_], 'C2', 'C13', 'gamma0'),
        ([0, a_], 'C3', 'C14', 'gamma0'),

        ([-a_, 0], 'C1', 'C14', 'gamma0'),  #
        ([-a_, 0], 'C6', 'C7', 'gamma0'),
        ([-a_, 0], 'C9', 'C12', 'gamma0'),

        # layer 2
        ([0, a_], 'C15', 'C16', 'gamma0'),
        ([0, a_], 'C16', 'C27', 'gamma0'),
        ([0, a_], 'C17', 'C28', 'gamma0'),

        ([-a_, 0], 'C15', 'C26', 'gamma0'),  #
        ([-a_, 0], 'C19', 'C22', 'gamma0'),
        ([-a_, 0], 'C25', 'C26', 'gamma0'),

        # not interlayer hopping
        ([0, 0], 'C1', 'C15', 'gamma1'),
    )
    lat.min_neighbors = 2
    return lat


def ring(inner_radius, outer_radius):
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)
    return pb.FreeformShape(contains, width=[2*outer_radius, 2*outer_radius])


if __name__ == '__main__':
    c0 = 0.335  # [nm] graphene interlayer spacing

    lattice = twist_bilayer_graphene()
    lattice.plot()
    plt.title('lattice')
    plt.show()
    shape = pb.circle(radius=20)
    model = pb.Model(lattice, shape)  #  pb.translational_symmetry())
    model.plot()
    plt.title('model')
    plt.show()
    # solver = pb.solver.lapack(model)

    # a_cc = 6.51172  # 1.42098
    # # a_ = 6.51172
    # Gamma = [0, 0]
    # K1 = [-4 * np.pi / (3 * 3 ** 0.5 * a_cc), 0]
    # M = [0, 2 * np.pi / (3 * a_cc)]
    # K2 = [2 * np.pi / (3 * 3 ** 0.5 * a_cc), 2 * np.pi / (3 * a_cc)]
    #
    # bands = solver.calc_bands(K1, Gamma, M, K2)
    # bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
    # plt.show()
    print('finish')