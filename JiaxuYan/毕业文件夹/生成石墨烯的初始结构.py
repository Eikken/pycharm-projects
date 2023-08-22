#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   生成石墨烯的初始结构.py    
@Time    :   2023/3/8 17:03  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    主要用pybinding生成
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pybinding as pb
from math import sqrt


def monolayer_graphene():
    a = 0.24595  # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8  # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a / 2, a / 2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc / 2]),
                        ('B', [0, a_cc / 2]))
    lat.add_hoppings(
        # inside the main cell
        ([0, 0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat


lattice = monolayer_graphene()
lattice.plot()
plt.show()
