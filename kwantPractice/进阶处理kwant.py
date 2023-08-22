#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   进阶处理kwant.py    
@Time    :   2022/1/11 16:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import tinyarray as tna
import kwant
import matplotlib.pyplot as plt
import numpy as np


# sigma_0 = numpy.array([[1, 0], [0, 1]])
# sigma_x = numpy.array([[0, 1], [1, 0]])
# sigma_y = numpy.array([[0, -1j], [1j, 0]])
# sigma_z = numpy.array([[1, 0], [0, -1]])
sigma_0 = tna.array([[1, 0], [0, 1]])
sigma_x = tna.array([[0, 1], [1, 0]])
sigma_y = tna.array([[0, -1j], [1j, 0]])
sigma_z = tna.array([[1, 0], [0, -1]])


def make_system(t=1.0, alpha=0.5, e_z=0.08, W=10, L=30):
    lat = kwant.lattice.square()
    syst = kwant.Builder()

    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t * sigma_0 + e_z * sigma_z
    # hoppings in x-direction
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t * sigma_0 + 1j * alpha * sigma_y / 2
    # hoppings in y-direction
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t * sigma_0 - 1j * alpha * sigma_x / 2

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t * sigma_0 + e_z * sigma_z
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = -t * sigma_0 + 1j * alpha * sigma_y / 2
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t * sigma_0 - 1j * alpha * sigma_x / 2
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst


def plot_conductance(syst, energies):
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))
    plt.figure()
    plt.plot(energies, data)
    plt.xlabel("energy [t]")
    plt.ylabel("conductance [e^2/h]")
    plt.show()


if __name__ == '__main__':
    syst = make_system()
    # kwant.plot(syst)
    syst = syst.finalized()
    plot_conductance(syst, energies=[i*0.01 for i in range(100)])

