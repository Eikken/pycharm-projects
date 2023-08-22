#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Eq15.py    
@Time    :   2022/1/10 0:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


import numpy as np
import matplotlib.pyplot as plt
import functools


def band_dispersion(a=2.46):
    ka = np.linspace(2*np.pi/3, 4*np.pi/3, 400)
    N = 400
    t = 1
    tl = 0.2 * t
    EK = np.zeros((N, 10))
    vL = [val for val in range(1, 11)]
    i0 = 0
    for i in vL:
        j0 = 0
        for j in ka:
            Dk = -2 * np.cos(j / 2)
            ek = i / 2 * ((1 + Dk ** 2) ** 2 - tl ** 2 / t ** 2) / ((1 + Dk ** 2) ** 2 + tl ** 2 / t ** 2)
            EK[j0, i0] = ek
            j0 += 1
        i0 += 1

    for p in range(10):
        plt.plot(ka, EK[:, p], '-k')
    plt.show()


def plot_bands_one_dimension(k, hamiltonian0):
    pass


if __name__ == '__main__':
    band_dispersion()

    # print(vL)