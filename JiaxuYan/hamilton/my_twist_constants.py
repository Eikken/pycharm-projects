#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   my_twist_constants.py
@Time    :   2022/8/15 10:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
from pybinding import hopping_energy_modifier
from pybinding.constants import hbar


a = 0.24595   #: [nm] unit cell length
a_cc = 0.142  #: [nm] carbon-carbon distance
t = -2.8  #: [eV] nearest neighbor hopping
t_nn = 0.1  #: [eV] next-nearest neighbor hopping
vf = 3 / (2 * hbar) * abs(t) * a_cc  #: [nm/s] Fermi velocity
beta = 3.37  #: strain hopping modulation

a_lat_21_7 = 6.51172  # supercell lattice length of 21.7 degree.
a_nn__21_7 = 1.42097
a_nnn_21_7 = 2.46120
a_t_21_7 = 6.70900  # 可以修改的参数值

_default_3band_params = {  # from https://doi.org/10.1103/PhysRevB.88.085433
    # ->           a,  eps1,  eps2,     t0,    t1,    t2,   t11,   t12,    t22
    "MoS2":  [0.3190, 1.046, 2.104, -0.184, 0.401, 0.507, 0.218, 0.338,  0.057],
    "WS2":   [0.3191, 1.130, 2.275, -0.206, 0.567, 0.536, 0.286, 0.384, -0.061],
    "MoSe2": [0.3326, 0.919, 2.065, -0.188, 0.317, 0.456, 0.211, 0.290,  0.130],
    "WSe2":  [0.3325, 0.943, 2.179, -0.207, 0.457, 0.486, 0.263, 0.329,  0.034],
    "MoTe2": [0.3557, 0.605, 1.972, -0.169, 0.228, 0.390, 0.207, 0.239,  0.252],
    "WTe2":  [0.3560, 0.606, 2.102, -0.175, 0.342, 0.410, 0.233, 0.270,  0.190],
}

hopping_energy = 1