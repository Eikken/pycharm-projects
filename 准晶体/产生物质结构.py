#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   产生物质结构.py    
@Time    :   2021/9/16 14:04  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import pybinding as pb

import matplotlib.pyplot as plt

from math import sqrt, pi

c0 = 0.335  # [nm] graphene interlayer spacing


def two_graphene_monolayers():
    """Two individual AB stacked layers of monolayer graphene without interlayer hopping"""
    from pybinding.repository.graphene.constants import a_cc, a, t

    lat = pb.Lattice(a1=[a/2, a/2 * math.sqrt(3)], a2=[-a/2, a/2 * math.sqrt(3)])
    lat.add_sublattices(('A1', [0,   a_cc,   0]),
                        ('B1', [0,      0,   0]),
                        ('A2', [0,      0, -c0]),
                        ('B2', [0,  -a_cc, -c0]))
    lat.register_hopping_energies({'gamma0': t})
    lat.add_hoppings(
        # layer 1
        ([0, 0], 'A1', 'B1', 'gamma0'),
        ([0, 1], 'A1', 'B1', 'gamma0'),
        ([1, 0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([0, 0], 'A2', 'B2', 'gamma0'),
        ([0, 1], 'A2', 'B2', 'gamma0'),
        ([1, 0], 'A2', 'B2', 'gamma0'),
        # not interlayer hopping
    )
    lat.min_neighbors = 2
    return lat



pb.pltutils.use_style()

d = 1  # [nm] unit cell length
t = 1    # [eV] hopping energy
rt3 = sqrt(3)

e1 = -0.699734+1.3285
delta1 = -0.027311
h0 = [[e1,delta1,delta1],[delta1,e1,delta1],[delta1,delta1,e1]]

t1_11 = -0.248774
t1_21 = -0.033678
t1_31 = -0.015000
t1_12 = -0.064970
t1_22 = 0.107549
t1_32 = 0.062797
t1_13 = -0.098159
t1_23 = 0.013446
t1_33 = 0.107787

T1_01 = [[t1_11,t1_12,t1_13],[t1_21,t1_22,t1_23],[t1_31,t1_32,t1_33]]
T1_10 = [[t1_33,t1_13,t1_23],[t1_31,t1_11,t1_21],[t1_32,t1_12,t1_22]]
T1_11 = [[t1_22,t1_23,t1_21],[t1_32,t1_33,t1_31],[t1_12,t1_13,t1_11]]

t2_11 = 0.024829
t2_21 = -0.071677
t2_31 = 0.002483
t2_12 = -0.014436
t2_22 = 0.018802
t2_32 = -0.009754
t2_13 = 0.003559
t2_23 = 0.028743
t2_33 = 0.025042

T2_m11 = [[t2_11,t2_12,t2_13],[t2_21,t2_22,t2_23],[t2_31,t2_32,t2_33]]
T2_21 = [[t2_33,t2_31,t2_32],[t2_13,t2_11,t2_12],[t2_23,t2_21,t2_22]]
T2_12 = [[t2_22,t2_32,t2_12],[t2_23,t2_33,t2_13],[t2_21,t2_31,t2_11]]

t3_11 = -0.076167
t3_21 = 0.002559
t3_31 = 0.005791
t3_12 = 0.006461
t3_22 = 0.002903
t3_32 = -0.010356
t3_13 = -0.012489
t3_23 = -0.008995
t3_33 = 0.012197

T3_02 = [[t3_11,t3_12,t3_13],[t3_21,t3_22,t3_23],[t3_31,t3_32,t3_33]]
T3_20 = [[t3_33,t3_13,t3_23],[t3_31,t3_11,t3_21],[t3_32,t3_12,t3_22]]
T3_22 = [[t3_22,t3_23,t3_21],[t3_32,t3_33,t3_31],[t3_12,t3_13,t3_11]]



# create a simple 2D lattice with vectors a1 and a2
lattice = pb.Lattice(a1=[d, 0], a2=[-0.5*d, d*rt3/2])
lattice.add_sublattices(
    ('A', [0, 0], h0)  # add an atom called 'A' at position [0, 0]
)
lattice.add_hoppings(
    # (relative_index, from_sublattice, to_sublattice, energy)
    ([0, 1],  'A', 'A', T1_01),
    ([1, 0],  'A', 'A', T1_10),
    ([1, 1],  'A', 'A', T1_11),


    ([1, -1], 'A', 'A', T2_m11),
    ([2, 1],  'A', 'A', T2_21),
    ([1,2],   'A', 'A', T2_12),

    ([0, 2], 'A', 'A', T3_02),
    ([2, 0], 'A', 'A', T3_20),
    ([2, 2], 'A', 'A', T3_22)
)
#lattice.plot()
#lattice.plot_brillouin_zone()


model = pb.Model(lattice, pb.translational_symmetry())
model.plot()
# solver = pb.solver.lapack(model)
# Gamma = [0, 0]
# M = [0, 2*pi/3]
# K = [2*pi/(3*sqrt(3)), 2*pi/3]
# bands = solver.calc_bands(Gamma, M, K, Gamma)
# bands.plot(point_labels=[r'$\Gamma$', 'M', 'K',r'$\Gamma$'])
plt.show()