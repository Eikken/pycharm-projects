#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   产生方格子.py    
@Time    :   2021/9/16 14:06  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   Polygon 创建2D形状，可用于困住石墨烯结构plot
'''
import pybinding as pb
from pybinding.repository import graphene
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


def asymmetric_strain(c):
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = -c / 2 * x ** 2 + c / 3 * x + 0.1
        uy = -c * 2 * x ** 2 + c / 4 * x
        return x + ux, y + uy, z

    return displacement


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


# lattice = monolayer_graphene()
# lattice.plot_brillouin_zone()
def rectangle(width, height):
    x0 = width / 2
    y0 = height / 2
    return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])


def trapezoid(a, b, h):
    return pb.Polygon([[-a / 2, 0], [-b / 2, h], [b / 2, h], [a / 2, 0]])


def circle(radius):
    def contains(x, y, z):
        return np.sqrt(x ** 2 + y ** 2) < radius

    return pb.FreeformShape(contains, width=[2 * radius, 2 * radius])


def ring(inner_radius, outer_radius):
    def contains(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2)
        return np.logical_and(inner_radius < r, r < outer_radius)

    return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])


#
# shape = ring(inner_radius=1.4, outer_radius=2)
# shape.plot()
rec = pb.rectangle(x=6, y=1)
hexa = pb.regular_polygon(num_sides=6, radius=1.92, angle=np.pi / 6)
cir = pb.circle(radius=0.6)
shape = rec + hexa - cir
model = pb.Model(
    graphene.monolayer(),
    # trapezoid(a=3.2, b=1.4, h=1.5)
    # circle(radius=2.5)
    shape
)
model.plot()
# model.shape.plot()
# model = pb.Model(
#     graphene.monolayer(),
#     pb.primitive(a1=6, a2=6)  # 在a1 a2方向上扩胞
# )
# model.plot()
plt.show()
# model = pb.Model(
#     graphene.bilayer(),
#     pb.regular_polygon(num_sides=3, radius=1.1),
#     asymmetric_strain(c=0.42)
# )
# model.plot()
# d = 0.2  # [nm] unit cell length
# t = 1    # [eV] hopping energy
#
# # create a simple 2D lattice with vectors a1 and a2
# lattice = pb.Lattice(a1=[d, 0], a2=[0, d])
# lattice.add_sublattices(
#     ('A', [0, 0])  # add an atom called 'A' at position [0, 0]
# )
# lattice.add_hoppings(
#     # (relative_index从主单元格开始到达另一个单元格所需的整数步骤数。
#     # , from_sublattice, to_sublattice,
#     # from_sublattice表示[0,0]单元中的子格，to_sublattice表示相邻单元中的子格。
#     # energy 跳跃能量的值)
#     ([0, 1], 'A', 'A', t),
#     ([1, 0], 'A', 'A', t)
# )
# lattice.plot()
# plt.show()
