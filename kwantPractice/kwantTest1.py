#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   kwantTest1.py    
@Time    :   2022/1/11 10:26  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import kwant
from matplotlib import pyplot


def make_system(a=1, t=1.0, W=10, L=30):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.square(a)
    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t
    syst[lat.neighbors()] = -t
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst


def plot_conductance(syst, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))
    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()


if __name__ == '__main__':
    syst = make_system()
    # Check that the system looks as intended.
    # kwant.plot(syst)
    syst = syst.finalized()
    plot_conductance(syst, energies=[0.01 * i for i in range(100)])
# if __name__ == '__main__':
#     syst = kwant.Builder()
#     a = 1
#     lat = kwant.lattice.square(a)
#     t = 1.0
#     W = 10
#     L = 30
#     for i in range(L):
#         for j in range(W):
#              # On-site Hamiltonian
#             syst[lat(i, j)] = 4 * t
#             # Hopping in y-direction
#             if j > 0:
#                 syst[lat(i, j), lat(i, j - 1)] = -t
#             # Hopping in x-direction
#             if i > 0:
#                 syst[lat(i, j), lat(i - 1, j)] = -t
#     sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
#     left_lead = kwant.Builder(sym_left_lead)
#     for j in range(W):
#         left_lead[lat(0, j)] = 4 * t
#         if j > 0:
#             left_lead[lat(0, j), lat(0, j - 1)] = -t
#         left_lead[lat(1, j), lat(0, j)] = -t
#     syst.attach_lead(left_lead)
#     sym_right_lead = kwant.TranslationalSymmetry((a, 0))
#     right_lead = kwant.Builder(sym_right_lead)
#     for j in range(W):
#         right_lead[lat(0, j)] = 4 * t
#         if j > 0:
#             right_lead[lat(0, j), lat(0, j - 1)] = -t
#         right_lead[lat(1, j), lat(0, j)] = -t
#     syst.attach_lead(right_lead)
#     kwant.plot(syst)
#
#     syst = syst.finalized()
#     energies = []
#     data = []
#     for ie in range(100):
#         energy = ie * 0.01
#         # compute the scattering matrix at a given energy
#         smatrix = kwant.smatrix(syst, energy)
#         # compute the transmission probability from lead 0 to
#         # lead 1
#         energies.append(energy)
#         data.append(smatrix.transmission(1, 0))
#     pyplot.figure()
#     pyplot.plot(energies, data)
#     pyplot.xlabel("energy [t]")
#     pyplot.ylabel("conductance [e^2/h]")
#     pyplot.show()