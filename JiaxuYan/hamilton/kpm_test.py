#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   kpm_test.py    
@Time    :   2022/8/22 15:56  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   calculate kpm \ ldos \ dos
            Kernel polynomial method 核多项式方法参照：https://doi.org/10.1103/RevModPhys.78.275
            kpm就是要根据big size supercell来计算ldos
            与arpack和lapack差不多
            spatial dos 必须要有shape才能计算
'''

import time

import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
import peakutils as pk
from pybinding.repository import graphene
from JiaxuYan.pbtest.genPosDataStructure import gen_lattice


def your_func_here(*args, **kwargs):
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


def calu_kpm_dos(**kwargs):
    broadening = 0.15
    lattice_ = kwargs['lat']
    shape = pb.circle(radius=500)
    this_model = pb.Model(lattice_,
                          # pb.translational_symmetry(),
                          shape,
                          )

    # this_model.plot()
    # this_model.lattice.plot_vectors(position=[0, 0])
    # plt.title('path model')
    # plt.show()
    # plt.clf()

    kpm = pb.kpm(this_model)

    eg = 6
    dos = kpm.calc_dos(energy=np.linspace(-eg, eg, 1000), broadening=broadening, num_random=100)
    dos.plot()
    indexes = pk.indexes(dos.data, thres=0.1, min_dist=50)
    # plt.scatter(dos.variable[indexes], dos.data[indexes], marker='*', color='red', lw=2)
    plt.title('%deV dos %.2f' % (eg, broadening))
    plt.yticks([])
    plt.show()
    plt.clf()
    print('peak data values: \n[eV, \tDOS]')
    for i, j in zip(dos.variable[indexes], dos.data[indexes]):
        print('[%.2f, %.2f]' % (i, j))


def calu_kpm_ldos(c28_, **kwargs):
    # lots of list C atoms.
    # 设置参数，选择不同的ldos for 28 atoms
    # ask question here >>

    broadening = 0.03
    lattice_ = kwargs['lat']
    shape = pb.circle(radius=100)
    shape1 = pb.rectangle(400, 400)
    this_model = pb.Model(lattice_,
                          # pb.translational_symmetry(),
                          # pb.regular_polygon(num_sides=3, radius=15, angle=np.pi),
                          shape1,
                          # asymmetric_strain(c=0.02)  # pull force
                          )

    this_model.plot()
    # this_model.lattice.plot_vectors(position=[0, 0])
    plt.title('path model')
    plt.show()
    plt.clf()

    kpm = pb.kpm(this_model)

    this_eV = 5

    ldos = kpm.calc_ldos(energy=np.linspace(-this_eV, this_eV, 500), broadening=broadening,
                         position=[0, 0])
    ldos.plot(label='111')
    pb.pltutils.legend()
    plt.title('ldos %.2f' % broadening)
    plt.show()
    plt.clf()
    # for sub_name in c28_:
    #     print(sub_name)
    #     ldos = kpm.calc_ldos(energy=np.linspace(-this_eV, this_eV, 500), broadening=broadening,
    #                          position=[0, 0], sublattice=sub_name)
    #     ldos.plot(label=sub_name)
    #     pb.pltutils.legend()
    #     plt.title('ldos %.2f' % broadening)
    #     plt.savefig('png/c28ldos/%deV ldos %s.png' % (this_eV, sub_name), dpi=100)
    #     plt.clf()
    # plt.show()
    # plt.clf()


def calu_spatial_ldos(*args, **kwargs):
    # 3D dos show
    broadening = 0.1

    lattice_ = kwargs['lat']
    shape = pb.circle(radius=20)
    this_model = pb.Model(lattice_,
                          shape,
                          # gaussian_bump_strain(height=1.6, sigma=7.5)
                          )

    this_model.plot()
    this_model.lattice.plot_vectors(position=[0, 0])
    plt.figure(figsize=(7, 4), dpi=100)
    plt.subplot(121, title="xy-plane")
    this_model.plot()
    plt.subplot(122, title="xz-plane")
    this_model.plot(axes="xz")
    plt.show()
    plt.clf()

    # model = pb.Model(graphene.monolayer(),
    #                  pb.regular_polygon(num_sides=6, radius=4.5),
    #                  gaussian_bump_strain(height=1.6, sigma=1.6))
    # kpm = pb.kpm(model)
    # spatial_ldos = kpm.calc_spatial_ldos(energy=np.linspace(-6, 6, 100), broadening=0.2,  # eV
    #                                      shape=pb.circle(radius=2.8))  # only within the shape
    # plt.figure(figsize=(6.7, 6))
    # gridspec = plt.GridSpec(2, 2, height_ratios=[1, 0.3], hspace=0)
    #
    # energies = [0.0, 0.75, 0.0, 0.75]  # eV
    # planes = ["xy", "xy", "xz", "xz"]
    #
    # for g, energy, axes in zip(gridspec, energies, planes):
    #     plt.subplot(g, title="E = {} eV, {}-plane".format(energy, axes))
    #     smap = spatial_ldos.structure_map(energy)
    #     smap.plot(site_radius=(0.02, 0.15), axes=axes)
    eg = 5
    kpm = pb.kpm(this_model)
    spatial_ldos = kpm.calc_spatial_ldos(energy=np.linspace(-eg, eg, 500), broadening=broadening,
                                         shape=pb.circle(radius=16))

    plt.figure(figsize=(9, 7), dpi=100)

    gridspec = plt.GridSpec(2, 3, hspace=0)  # height_ratios=[1, 1], hspace=0)

    energies = [0.0, 0.71, 1.24, 2.30, 2.80, 3.38]  # eV
    planes = [i for i in range(len(energies))]

    for g, energy, axes in zip(gridspec, energies, planes):
        plt.subplot(g, title="E = {} eV, {}-plane".format(energy, axes))
        smap = spatial_ldos.structure_map(energy)
        smap.plot(site_radius=(0.25, 0.45), axes='xy')

    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    lattice, all_hopping_acc = gen_lattice()

    c28_list = sorted(set([l for i in all_hopping_acc for j in i for l in j]), key=lambda a: int(a[1:]))
    # this_sort和lambda都可以重写排序

    calu_kpm_dos(lat=lattice)
    # calu_kpm_ldos(c28_list, lat=lattice)
    # calu_spatial_ldos(lat=lattice)

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
