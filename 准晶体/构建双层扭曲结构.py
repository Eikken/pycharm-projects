#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   构建双层扭曲结构.py    
@Time    :   2021/9/21 8:52
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import pybinding as pb
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt

c0 = 0.335  # [nm] 层间距


def two_graphene_monolayers():
    # AB stacked 的两个单层
    from pybinding.repository.graphene.constants import a_cc, a, t
    lat = pb.Lattice(
        a1=[a / 2, a / 2 * math.sqrt(3)],
        a2=[-a / 2, a / 2 * math.sqrt(3)]
    )
    lat.add_sublattices(('A1', [0, a_cc, 0]),
                        ('B1', [0, 0, 0]),
                        ('A2', [0, 0, -c0]),
                        ('B2', [0, -a_cc, -c0]))
    lat.register_hopping_energies({'gamma0': t})
    lat.add_hoppings(
        # layer1
        ([0, 0], 'A1', 'B1', 'gamma0'),
        ([0, 1], 'A1', 'B1', 'gamma0'),
        ([1, 0], 'A1', 'B1', 'gamma0'),
        # layer2
        ([0, 0], 'A2', 'B2', 'gamma0'),
        ([0, 1], 'A2', 'B2', 'gamma0'),
        ([1, 0], 'A2', 'B2', 'gamma0'),
    )
    lat.min_neighbors = 1
    return lat


def twist_layers(angle):
    theta = np.deg2rad(angle)

    # 产生一个AB堆叠的旋转图层
    @pb.site_position_modifier
    def rotate(x, y, z):
        layer2 = (z < 0)
        x0 = x[layer2]
        y0 = y[layer2]
        # x[layer2] = x0 * math.cos(theta) - y0 * math.sin(theta)
        # y[layer2] = y0 * math.cos(theta) + x0 * math.sin(theta)
        Matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # print(np.array([x[layer2], y[layer2]]))
        twistMatrix = np.dot(Matrix, np.array([x[layer2], y[layer2]]))
        # print(twistMatrix)
        x[layer2] = twistMatrix[0, :]
        y[layer2] = twistMatrix[1, :]
        return x, y, z

    @pb.hopping_generator('interlayer', energy=0.1)  # eV
    def interlayer_generator(x, y, z):
        # 生成点与点之间的距离
        position = np.stack([x, y, z], axis=1)
        layer1 = (z == 0)
        layer2 = (z != 0)
        d_min = c0 * 0.99
        d_max = c0 * 1.01
        kdtree1 = cKDTree(position[layer1])
        kdtree2 = cKDTree(position[layer2])
        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')
        idx = coo.data > d_min
        abs_idx1 = np.flatnonzero(layer1)
        abs_idx2 = np.flatnonzero(layer2)
        row, col = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]
        return row, col

    @pb.hopping_energy_modifier
    def interlayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
        # 将新生成的约束关系转化为距离的函数
        d = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        interlayer = (hop_id == 'interlayer')
        energy[interlayer] = 0.4*c0/d[interlayer]
        return energy
    return rotate, interlayer_generator, interlayer_hopping_value

if __name__ == '__main__':
    model = pb.Model(
        two_graphene_monolayers(),
        pb.circle(radius=1.6),
        twist_layers(angle=6.01)
    )
    plt.figure(figsize=(6.5, 6.5))
    model.plot()
    # print(model.onsite_map.xyz)
    plt.title('angle = 6.01 °')
    plt.show()
    print(model.hamiltonian)
    print('finish')