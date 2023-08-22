#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   sitmap_test.py
@Time    :   2022/8/24 9:32  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time

# import math
import matplotlib.pyplot as plt
import pybinding as pb
import pandas as pd
import numpy as np
import peakutils as pk
from pybinding.repository import graphene
from JiaxuYan.pbtest.genPosDataStructure import gen_lattice


def your_func_here(*args, **kwargs):
    pass


def wavy(a, b):
    @pb.onsite_energy_modifier
    def potential(x, y):
        return np.sin(a * x)**2 + np.cos(b * y)**2
    return potential


def linear(k):
    @pb.onsite_energy_modifier
    def potential(energy, x):
        return energy + k*x
    return potential


def wavy2(a, b):
    @pb.onsite_energy_modifier
    def potential(energy, x, y):
        v = np.sin(a * x)**2 + np.cos(b * y)**2
        return energy + v
    return potential


def structureMap_plot_show(*args, **kwargs):
    lattice_ = kwargs['lat']

    shape1 = pb.circle(radius=12)
    this_model = pb.Model(
        lattice_,
        shape1,
        # wavy(a=0.6, b=0.9),
        wavy2(a=0.6, b=0.9),
        linear(k=0.2),

    )
    this_model.plot()
    plt.show()
    plt.clf()

    this_model.onsite_map.plot_contourf()
    pb.pltutils.colorbar(label="U (eV)")

    plt.show()
    plt.clf()


if __name__ == '__main__':
    time1 = time.time()
    # write here

    lattice, all_hopping_acc = gen_lattice()

    structureMap_plot_show(lat=lattice)

    time2 = time.time()

    print('>> Finished, use time %d s' % (time2 - time1))
