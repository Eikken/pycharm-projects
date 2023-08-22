#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   StrainTest.py    
@Time    :   2022/8/15 10:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   plotly
            Parameters  ----------
            num_sides : int
             Number of sides.
            radius : float
             Radius of the circle which connects all the vertices of the polygon.
            angle : float
             Rotate the polygon.

             calu_spatial_ldos()  计算局域态密度 Calculate the LDOS as a function of energy
                                  and space (in the area of the given shape).
             calu_periodic_model()  周期模型 k_path路径计算
             gaussian_bump_strain（） 增加高斯模型束缚
'''

import time

# import math
import matplotlib.pyplot as plt
import pybinding as pb
# import pandas as pd
import numpy as np
from pybinding.repository import graphene

from JiaxuYan.hamilton import my_twist_constants

from JiaxuYan.pbtest.genPosDataStructure import gen_lattice


def this_sort(item):
    return int(item[1:])


def calu_periodic_model(*args, **kwargs):
    model_ = pb.Model(lattice,
                      pb.translational_symmetry(),
                      # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
                      # shape,
                      )
    solver = pb.solver.lapack(model_)  # 注意 lapack 和 arpack有所区分和不同，计算量大小等……
    a_cc = my_twist_constants.a_lat_21_7 / 3 ** 0.5
    kx_lim = np.pi / a_cc
    kx_path = np.linspace(-kx_lim, kx_lim, 500)
    ky_outer = 0
    ky_inner = 2 * np.pi / (3 * a_cc)

    plt.figure(dpi=200)

    outer_bands = []
    for kx in kx_path:
        solver.set_wave_vector([kx, ky_outer])
        outer_bands.append(solver.eigenvalues)

    inner_bands = []
    for kx in kx_path:
        solver.set_wave_vector([kx, ky_inner])
        inner_bands.append(solver.eigenvalues)

    # for bands in [outer_bands, inner_bands]:
    #     result = pb.results.Bands(kx_path, bands)
    #     result.plot(lw=0.75)
    result = pb.results.Bands(kx_path, outer_bands)
    result.plot(lw=0.75)
    hh = 5
    plt.ylim(-hh, hh)
    plt.title('periodic band plot')
    plt.show()

    result = pb.results.Bands(kx_path, inner_bands)
    result.plot(lw=0.75)
    hh = 5
    plt.ylim(-hh, hh)
    plt.title('periodic band plot')
    plt.show()


def solver_arpack(**kwargs):
    pass


def gaussian_bump_strain(height, sigma):
    """Out-of-plane deformation (bump)"""

    @pb.site_position_modifier
    def displacement(x, y, z):
        dz = height * np.exp(-(x ** 2 + y ** 2) / sigma ** 2)  # gaussian
        return x, y, z + dz  # only the height changes

    @pb.hopping_energy_modifier
    def strained_hoppings(energy, x1, y1, z1, x2, y2, z2):
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)  # strained neighbor distance
        return energy * np.exp(-3.37 * (d / graphene.a_cc - 1))  # see strain section

    return displacement, strained_hoppings


def calu_spatial_ldos(**kwargs):
    lattice_ = kwargs['lat']

    # @see Tips
    model_ = pb.Model(lattice_,  # graphene.monolayer().with_offset([-graphene.a / 2, 0]),
                      pb.regular_polygon(num_sides=6, radius=15),
                      gaussian_bump_strain(height=1.6, sigma=7.5))

    # plt.figure(figsize=(6.7, 2.2), dpi=200)
    # plt.subplot(121, title="xy-plane")
    # # plt.subplot(121, title="xy-plane", ylim=[-5, 5])
    # model_.plot()
    # plt.subplot(122, title="xz-plane")
    # model_.plot(axes="xz")
    #
    # plt.show()
    kpm = pb.kpm(model_)
    spatial_ldos = kpm.calc_spatial_ldos(energy=np.linspace(-3, 3, 100), broadening=0.2,  # eV
                                         shape=pb.circle(radius=2.8))  # only within the shape

    gridspec = plt.GridSpec(2, 2, height_ratios=[1, 0.3], hspace=0)
    energies = [0.0, 0.75, 0.0, 0.75]  # eV
    planes = ["xy", "xy", "xz", "xz"]

    for g, energy, axes in zip(gridspec, energies, planes):
        plt.subplot(g, title="E = {} eV, {}-plane".format(energy, axes))
        smap = spatial_ldos.structure_map(energy)
        smap.plot(site_radius=(0.02, 0.15), axes=axes)

    plt.show()


def asymmetric_strain(c):
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = -c / 2 * x ** 2 + c / 3 * x + 0.1
        uy = -c * 2 * x ** 2 + c / 4 * x
        return x + ux, y + uy, z

    return displacement


if __name__ == '__main__':
    time1 = time.time()
    # write here
    lattice, all_hopping_acc = gen_lattice()
    shape = pb.circle(radius=15)
    model = pb.Model(lattice,
                     # pb.translational_symmetry(),
                     # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
                     shape,
                     # asymmetric_strain(c=0.02)  # 拉拽的力
                     )
    broadening = 0.03

    model.plot()
    # model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
    plt.title('model')
    plt.show()

    # c28_list = sorted(set([l for i in all_hopping_acc for j in i for l in j]), key=this_sort)

    calu_periodic_model()
    # calu_ldos(c28_list, lat=lattice)

    model = pb.Model(lattice,
                     pb.translational_symmetry(),
                     # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
                     # shape,
                     )
    # plt.figure(figsize=(7, 2.5), dpi=300)
    # grid = plt.GridSpec(nrows=1, ncols=2)
    # for block, energy in zip(grid, [0, 0.25]):
    #     plt.subplot(block)
    #     plt.title("E = {} eV".format(energy))
    #
    #     solver = pb.solver.arpack(model, k=30, sigma=energy)  # for the 30 lowest energy eigenvalues
    #     ldos_map = solver.calc_spatial_ldos(energy=energy, broadening=broadening)
    #     ldos_map.plot(site_radius=(0.25, 0.25))
    # plt.show()

    # solver = pb.solver.arpack(model, k=10)  # for the 20 lowest energy eigenvalues
    # eigenvalues = solver.calc_eigenvalues(map_probability_at=[0.1, 0.6])
    # # eigenvalues.plot()
    # eigenvalues.plot_heatmap(show_indices=True)
    # pb.pltutils.colorbar()
    # plt.show()
    # sitNum = 24
    # probability_map = solver.calc_probability(sitNum)
    # probability_map.plot(site_radius=0.25)
    # plt.title('around {}'.format(sitNum))
    # plt.show()
    #
    # ldos_map = solver.calc_spatial_ldos(energy=0, broadening=0.05)  # [eV]
    # ldos_map.plot(site_radius=0.25)
    # plt.show()
    #
    # dos = solver.calc_dos(energies=np.linspace(-1, 1, 200), broadening=0.05)  # [eV]
    # dos.plot()
    # plt.show()

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
