#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   石墨烯单层.py    
@Time    :   2021/9/18 16:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

"""Several finite-sized systems created using builtin lattices and shapes"""
import pybinding as pb
from pybinding.repository import graphene
import matplotlib.pyplot as plt
from math import pi

pb.pltutils.use_style()

model_1 = pb.Model(
    graphene.monolayer(),
    # pb.rectangle(x=2, y=1.2)
    pb.regular_polygon(num_sides=6, radius=1.4)
)
model_2 = pb.Model(
    graphene.bilayer(),
    graphene.hexagon_ac(side_width=1)
)

model_3 = pb.Model(
    graphene.monolayer(),
    # pb.rectangle(x=2, y=1.2)
    pb.regular_polygon(num_sides=6, radius=1.4, angle=0)
)
model_2.plot()
# model_2.plot()
plt.show()