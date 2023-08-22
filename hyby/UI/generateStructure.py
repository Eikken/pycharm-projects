#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   generateStructure.py
@Time    :   2021/9/21 8:52
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import pybinding as pb
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene

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
    lat.min_neighbors = 2
    return lat


def twist_layers(angle=0.0):
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
        d_min = c0 * 0.98
        d_max = c0 * 1.1
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
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        interlayer = (hop_id == 'interlayer')
        energy[interlayer] = 0.4 * c0 / d[interlayer]
        return energy

    return rotate, interlayer_generator, interlayer_hopping_value


def hbn_layer(shape):
    """Generate hBN layer defined by the shape with intralayer hopping terms"""
    from pybinding.repository.graphene.constants import a_cc, t

    a_bn = 56 / 55 * a_cc  # ratio of lattice constants is 56/55

    vn = -1.4  # [eV] nitrogen onsite potential
    vb = 3.34  # [eV] boron onsite potential

    def hbn_monolayer():
        """Create a lattice of monolayer hBN """

        a = math.sqrt(3) * a_bn
        lat = pb.Lattice(a1=[a / 2, a / 2 * math.sqrt(3)], a2=[-a / 2, a / 2 * math.sqrt(3)])
        lat.add_sublattices(('Br', [0, -a_bn, -c0], vb),
                            ('N', [0, 0, -c0], vn))

        lat.min_neighbors = 2  # no need for hoppings lattice is used only to generate coordinates
        return lat

    model = pb.Model(
        hbn_monolayer(),
        shape
    )

    subs = model.system.sublattices
    idx_b = np.flatnonzero(subs == model.lattice.sublattices["Br"].alias_id)
    idx_n = np.flatnonzero(subs == model.lattice.sublattices["N"].alias_id)
    positions_boron = model.system[idx_b].positions
    positions_nitrogen = model.system[idx_n].positions

    @pb.site_generator(name='Br', energy=vb)  # onsite energy [eV]
    def add_boron():
        """Add positions of newly generated boron sites"""
        return positions_boron

    @pb.site_generator(name='N', energy=vn)  # onsite energy [eV]
    def add_nitrogen():
        """Add positions of newly generated nitrogen sites"""
        return positions_nitrogen

    @pb.hopping_generator('intralayer_bn', energy=t)
    def intralayer_generator(x, y, z):
        """Generate nearest-neighbor hoppings between B and N sites"""
        positions = np.stack([x, y, z], axis=1)
        layer_bn = (z != 0)

        d_min = a_bn * 0.98
        d_max = a_bn * 1.1
        kdtree1 = cKDTree(positions[layer_bn])
        kdtree2 = cKDTree(positions[layer_bn])
        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max, output_type='coo_matrix')

        idx = coo.data > d_min
        abs_idx = np.flatnonzero(layer_bn)

        row, col = abs_idx[coo.row[idx]], abs_idx[coo.col[idx]]
        return row, col  # lists of site indices to connect

    @pb.hopping_energy_modifier
    def intralayer_hopping_value(energy, x1, y1, z1, x2, y2, z2, hop_id):
        """Set the value of the newly generated hoppings as a function of distance"""
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        intralayer = (hop_id == 'intralayer_bn')
        energy[intralayer] = 0.1 * t * a_bn / d[intralayer]
        return energy

    return add_boron, add_nitrogen, intralayer_generator, intralayer_hopping_value


def drawFigHBN(angle=0.0):
    shape = pb.circle(radius=2),
    agl = angle
    model = pb.Model(
        graphene.monolayer_alt(),  # reference stacking is AB (theta=0)
        shape,
        hbn_layer(shape=shape),
        twist_layers(agl),
    )
    plt.figure(figsize=(6.8, 7.5))
    s = model.system
    plt.subplot(2, 2, 1, title="graphene")
    s[s.z == 0].plot()
    plt.subplot(2, 2, 2, title="hBN")
    s[s.z < 0].plot()
    plt.subplot(2, 2, (3, 4), title="graphene/hBN")
    s.plot()
    plt.show()


def drawFigGraphene(angle=0.0):
    model = pb.Model(
        two_graphene_monolayers(),
        pb.circle(radius=1.7),
        twist_layers(angle=angle)
    )
    plt.figure(figsize=(6.5, 6.5))
    model.plot()
    plt.title(r"$\theta$ = 21.798 $\degree$")
    plt.show()


def saveFile_cif(angle=0.0):
    # 将结构坐标保存为.cif文件可供MS读取
    from pybinding.repository.graphene.constants import a_cc, a, t
    strText = "# generated by Yan Jiaxu's group.\n\n"
    model_sys = pb.Model(
        two_graphene_monolayers(),
        pb.circle(radius=1.5),
        twist_layers(angle=angle)
    )
    wdt = model_sys.shape.width
    strText += "%d atoms\n" % len(model_sys.system.xyz)
    strText += "2 atom types\n\n"
    strText += "%7.3f %7.3f xlo xhi\n" % (0.0, a)
    strText += "%7.3f %7.3f ylo yhi\n" % (0.0, a)
    strText += "%7.3f %7.3f zlo zhi\n" % (-c0, 0.0)
    strText += "%7.3f %7.3f %7.3f xy xz yz\n" % (a_cc, 0.0, 0.0)
    strText += '  Masses\n\n'
    strText += "1 12.011\n2 12.011\nAtoms\n\n"
    count = 1
    for i in model_sys.system.xyz[model_sys.system.z==0]:
        strText += "%d 1 %7.3f %7.3f %7.3f\n" % (count, i[0], i[1], i[2])
        count += 1
    for i in model_sys.system.xyz[model_sys.system.z<0]:
        strText += "%d 2 %7.3f %7.3f %7.3f\n" % (count, i[0], i[1], i[2])
        count += 1
    return strText
    # model_sys[model_sys.z < 0].x, model_sys[model_sys.z < 0].y
    # 当前system获取到整个结构的关键信息，包含siteMap等等，单层坐标获取如上述示例


if __name__ == '__main__':
    # drawFigHBN(21.787)
    # drawFigGraphene(21.787)
    # saveFile_cif(angle=21.787)
    model_sys = pb.Model(
        two_graphene_monolayers(),
        pb.circle(radius=1.7),
        twist_layers(angle=21.787)
    )
    print()
    # print(model_sys.xyz[:10])
