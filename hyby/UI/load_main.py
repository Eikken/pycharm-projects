#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   load_main.py
@Time    :   2021/9/26 13:33
@E-mail  :   iamwxyoung@qq.com
@Tips    :   先生成石墨烯结构，然后zip打包[xList,yList]，进行M(θ)角度的旋转并进行plt.show()
        min.(axis = )none：整个矩阵; 0：每列; 1：每行
'''

from __future__ import unicode_literals
import sys
import os

from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QApplication, QMessageBox, QFileDialog
from PySide2.QtUiTools import QUiLoader
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg  # pyqt5的画布
matplotlib.use('Qt5Agg')
from PySide2 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import logging
from logging import handlers

import random
import math
import pybinding as pb
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from scipy.spatial import distance
import xlwt

c0 = 0.335  # [nm] 层间距

class ShareInfo:
    wm = None
    showTest = None
    aw = None
    sc = None
    dilg = None
    myplt = None
    mySize = 20
    agl = 0.0

class Circle:
    x = 0.0
    y = 0.0
    r = 1.0

    def __init__(self, x, y, R):
        self.x = float(x)
        self.y = float(y)
        self.r = float(R)

    def calArea(self):
        return np.pi * self.r ** 2


# 求两圆相交的面积
def calShadow(circle1, circle2):
    d = ((circle1.x - circle2.x) ** 2 + (circle1.y - circle2.y) ** 2) ** 0.5
    if d == 0:
        return circle1.calArea()
    if d > 14:
        print('[', circle1.x, ',', circle1.y, '] 和 [', circle2.x, ',', circle2.y, ']不重叠')
        return 0.0
    else:
        ang1 = np.arccos((circle1.r ** 2 + d ** 2 - circle2.r ** 2) / 2.0 / circle1.r / d)
        ang2 = np.arccos((-circle1.r ** 2 + d ** 2 + circle2.r ** 2) / 2.0 / circle2.r / d)
        area = ang1 * circle1.r ** 2 + ang2 * circle2.r ** 2 - d * circle1.r * np.sin(ang1)
        return area


def calTotal(initMox, bs=100):
    circle = Circle(0, 0, 0.07 * bs)
    pointsNum = len(initMox)
    total_area = pointsNum * circle.calArea()
    return total_area


# 求list中圆相交的总面积
def sumArea(set1, set2, bs=100):
    emptyList = []
    for s in set1:
        minDistance = distance.cdist([s], set2, 'euclidean').min(axis=1)
        indexTuple = np.where(distance.cdist(set2, [s], 'euclidean') == minDistance)
        # set2[indexTuple[0]][0]是准确的数据
        result = calShadow(Circle(s[0], s[1], 0.07 * bs),
                           Circle(set2[indexTuple[0]][0][0], set2[indexTuple[0]][0][1], 0.07 * bs))
        emptyList.append(result)
    return sum(emptyList)


def genGraphene(Super=10, bs=1):  # 返回新的坐标的大胞
    # 原胞中C的坐标
    a = (2.460, 0, 0)
    b = (2.460 / 2, 2.460 / 2 * math.sqrt(3), 0)
    c = (0, 0, 20)
    # 扩胞矩阵
    super_x = Super
    super_y = Super
    super_z = 1

    extendCellMatrix = np.array([[super_x, 0, 0],
                                 [0, super_y, 0],
                                 [0, 0, super_z]])
    lattice = np.array([a, b, c])
    # 矩阵右乘扩胞矩阵3X3 * 3X3，生成新的大胞
    extendLattice = np.dot(lattice, extendCellMatrix)
    # C1 = [0, 0, 0.5]
    # C2 = [1 / float(3), 1 / float(3), 0.5]
    Frac1 = 0
    Frac2 = 1 / float(3)
    allAtoms = []
    for i in range(super_x):
        for j in range(super_y):
            newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.5]
            newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.5]
            allAtoms.append(newC1)
            allAtoms.append(newC2)
    newAllAtoms = np.dot(np.array(allAtoms), extendLattice)
    x_List = np.array(newAllAtoms).T[0] * bs
    y_List = np.array(newAllAtoms).T[1] * bs
    z_List = np.array(newAllAtoms).T[2] * bs
    x_Mean = np.mean(x_List)
    y_Mean = np.mean(y_List)
    x_List = x_List - np.mean(x_List)
    y_List = y_List - np.mean(y_List)
    return x_List, y_List, z_List, x_Mean, y_Mean


def matrixTransformation(x_, y_, theta):
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    xT, yT = [], []
    for k, v in zip(x_, y_):
        twistMatrix = np.dot([k, v], Matrix)
        # 矩阵 1X2 * 2X2 = 1X2
        xT.append(twistMatrix[0])
        yT.append(twistMatrix[1])
    return np.array(xT), np.array(yT)


# 画圆函数
def f(x, R):
    return (R ** 2 - x ** 2) ** 0.5


# 计算(x,y)和原点(0,0)的距离
def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def doublePointsDistance(ix, iy, tx, ty):
    return ((tx - ix) ** 2 + (ty - iy) ** 2) ** 0.5


def overFlowDrop(xL, yL, R):
    xDrop = np.delete(xL, np.where(xL.__abs__() > R))  #
    yDrop = np.delete(yL, np.where(xL.__abs__() > R))
    return xDrop, yDrop


# initXY是未旋转的初始坐标集合，twistXY是旋转后的坐标集合，两者计算overlap

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def drawFig(x1, y1, x2, y2, angleTheta, r, set1):
    xIndex = np.linspace(-r, r, int(r))
    plt.figure(figsize=(8, 8), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x1, y1, 60, marker='.', color='green')
    plt.scatter(x2, y2, 60, marker='.', color='blue')
    plt.plot(xIndex, f(xIndex, r), lw=1, color='red')
    plt.plot(xIndex, -f(xIndex, r), lw=1, color='red')
    plt.scatter(set1[:, 0], set1[:, 1], 40, marker='*', color='red')
    plt.scatter(0, 0, 30, marker='*', color='black')
    # plt.savefig('png/aa_%.3f°.png' % angleTheta, dpi=500)
    print('showed , saved area_twist_%.3f°.png' % angleTheta)
    plt.show()


def drawOverLap(set1, set2, set3, angleTheta):
    plt.figure(figsize=(10, 10), edgecolor='black')
    # plt.subplot(111)
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    # kdtree1 = cKDTree(set1)
    # kdtree2 = cKDTree(set2)
    # coo = kdtree1.sparse_distance_matrix(kdtree2, 3.35, output_type='coo_matrix')
    plt.scatter(set1[:, 0], set1[:, 1], color='blue')
    plt.scatter(set2[:, 0], set2[:, 1], color='green')
    plt.scatter(set3[:, 0], set3[:, 1], 50, marker='*', color='red')
    plt.scatter(0, 0, 10, marker='*', color='black')
    plt.show()
    # plt.savefig('png/over_lap_%.2f°.png' % angleTheta, dpi=500)
    # print('showed, saved over_lap_%.2f°.png' % angleTheta)


def calEuclidean(s_1, s_2):
    # s1 为列标，s2为行标，求s2内的点到s1中每个点最近的，就得取行最小值。
    dis1 = distance.cdist(s_1, s_2, 'euclidean').min(axis=1)
    dis2 = distance.cdist(s_1, s_2, 'euclidean').min(axis=0)
    index_S1 = np.where(dis1 < 3)
    index_S2 = np.where(dis2 < 3)
    # df = pd.DataFrame(distance.cdist(s1, s2, 'euclidean')) # 数据转Excel
    # df.to_excel('data/%.3f°distance.xlsx'%angle, index=True, header=True)
    return index_S1, index_S2


def calSelfDistance(s_1, s_2, s_3, s_4, a):
    # s1 为列标，s2为行标，求s2内的点到s1中每个点最近的，就得取行最小值。
    # SumArea写在此处
    # print('overlap atoms:', s)
    # print('all atoms:', len(s_1))
    # print('ratio:', (6 / len(s_1)) * 100, '%')  # 这两个ratio其实是相似的
    # print('ratio:', (s / t) * 100, '%')
    s = sumArea(s_3, s_4, bs=100)
    t = calTotal(s_1, bs=100)
    ratio = 100 * s / t
    # resultDict[str(a)].append(s)
    # resultDict[str(a)].append('%.6f' % ratio + '%')
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    tmpDict1, tmpDict2 = {}, {}
    for h, j in zip(dis1, dis2):
        if int(h) in tmpDict1:
            tmpDict1[int(h)] += 1
        else:
            tmpDict1[int(h)] = 1
        if int(j) in tmpDict2:
            tmpDict2[int(j)] += 1
        else:
            tmpDict2[int(j)] = 1
    for h in tmpDict1.keys():
        tmpDict1[h] = np.where(np.trunc(dis1) == h)[0]
    for h in tmpDict2.keys():
        tmpDict2[h] = np.where(np.trunc(dis2) == h)[0]
    # drawArrow(s_1, s_2, tmpDict1, tmpDict2, a)
    distanceDict = {}
    for k1, k2 in zip(tmpDict1.keys(), tmpDict2.keys()):
        X1 = s_1[tmpDict1[k1][0]]
        X2 = s_2[tmpDict2[k2][0]]
        distanceDict[k1] = [calDoublePoint(X1, X2), len(tmpDict1[k1])]
        # 存储形式：{k:[[number, calDistance(point1,point2), [point1.x, point1.y],[point2.x, point2.y]],···]


def calAllDistance(s_1, s_2, cL, a):
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    index_S3, index_S4 = calEuclidean(s_1, s_2)
    tmpS1 = s_1[index_S3]
    tmpS2 = s_2[index_S4]
    dis3 = distance.cdist(tmpS1, [[0, 0]], 'euclidean').min(axis=1)
    dis4 = distance.cdist(tmpS2, [[0, 0]], 'euclidean').min(axis=1)
    # index_S1 = np.where(dis1 <= r)  # 加70作为胞半径
    # index_S2 = np.where(dis2 <= r)
    # index_S5 = np.where(dis3 < r)
    # index_S6 = np.where(dis4 < r)
    index_S1 = np.where(dis1 <= cL[str(a)] + 7)  # 加70作为胞半径
    index_S2 = np.where(dis2 <= cL[str(a)] + 7)
    index_S5 = np.where(dis3 < cL[str(a)] + 7)
    index_S6 = np.where(dis4 < cL[str(a)] + 7)
    outS1, outS2, outS3, outS4 = s_1[index_S1[0]], s_2[index_S2[0]], tmpS1[index_S5], tmpS2[index_S6]
    # out 1 2 是所有原子坐标，3 4是重叠的SuperCell坐标
    # return outS1, outS2, outS3, outS4
    calSelfDistance(outS1, outS2, outS3, outS4, a)


def calDoublePoint(param, param1):
    return ((param[0] - param1[0]) ** 2 + (param[1] - param1[1]) ** 2) ** 0.5


def drawArrow(s_1, s_2, s_3, tD1, tD2, a):
    plt.figure(figsize=(9, 9))
    plt.xticks([])
    plt.yticks([])
    for k1, k2 in zip(tD1.keys(), tD2.keys()):
        X1 = s_1[tD1[k1]]
        X2 = s_2[tD2[k2]]
        c1 = randomcolor()
        for i in range(len(X1)):
            j = i
            plt.arrow(X1[j:j + 1][0][0],
                      X1[j:j + 1][0][1],
                      X2[j:j + 1][0][0] - X1[j:j + 1][0][0],
                      X2[j:j + 1][0][1] - X1[j:j + 1][0][1],
                      width=0.5, head_width=20, head_length=20, overhang=0.9, color=c1)
    # plt.scatter(s_1[:, 0], s_1[:, 1], color=randomcolor())
    # plt.scatter(s_2[:, 0], s_2[:, 1], color=randomcolor())
    plt.plot(s_3[:, 0], s_3[:, 1], linestyle='--', color=randomcolor())
    # plt.scatter([0], [0], 20, marker='*', color='black')
    # plt.savefig('png/arrow/%.2farrow.png' % a, dpi=800)
    plt.show()
    # # sortIndex1 = np.argsort(dis1)
    # # sortIndex2 = np.argsort(dis2)
    # # df2 = pd.DataFrame(dis2)
    # # df2.to_excel('data/%.2fdf2.xls' % a, index=True, header=True)

def calDistance(s_1, s_2, r=0.0):
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    index_S3, index_S4 = calEuclidean(s_1, s_2)
    tmpS1 = s_1[index_S3]
    tmpS2 = s_2[index_S4]
    dis3 = distance.cdist(tmpS1, [[0, 0]], 'euclidean').min(axis=1)
    dis4 = distance.cdist(tmpS2, [[0, 0]], 'euclidean').min(axis=1)
    index_S1 = np.where(dis1 <= r)
    index_S2 = np.where(dis2 <= r)
    index_S5 = np.where(dis3 < r)
    index_S6 = np.where(dis4 < r)
    outS1, outS2, outS3, outS4 = s_1[index_S1[0]], s_2[index_S2[0]], tmpS1[index_S5], tmpS2[index_S6]
    return outS1, outS2, outS3, outS4


def calSuperCell(set1):
    # dis_list = set1[:2] # 失败
    dis1 = distance.cdist(set1, [(0, 0)], metric='euclidean')
    minDistance = dis1.min(axis=0)
    index = np.where(dis1 <= minDistance + 7)  # 为什么加7？考虑到原子半径
    super_cell = set1[index[0]]

    dis_set = distance.cdist(super_cell[1:], np.array(super_cell[0]), metric='euclidean')  # 计算出第一个点到集合内各点的距离，
    # 去除自身距离0
    minCell = dis_set.min()
    lenCell = minCell / np.tan(np.pi / 6.0)  # a / c = tan(π/6); 2 * c = 2 * (a / tan(π/6))
    return minCell, lenCell


def getDicts():
    resultDict = {'6.01': [1354.862355], '7.34': [1109.275439], '9.43': [863.9236077], '10.42': [1354.8623546323813],
                  '11.64': [1213.4891841297963], '13.17': [619.0864237], '15.18': [931.3409687], '16.43': [994.1971635],
                  '17.9': [790.7793624], '21.79': [375.771207], '24.43': [1162.550644], '26.01': [1262.3739541039336],
                  '27.8': [512.0898359], '29.41': [1398.815212957022]}
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359, '29.41': 1398.815212957022}
    return resultDict, cellLength

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


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


class MyMplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, dpi=400, angle=0.0):
        self.angle = angle
        self.Size = ShareInfo.mySize
        # 创建一个Figure,该Figure为matplotlib下的Figure，不是matplotlib.pyplot下面的Figure
        # 这里还要注意，width, heigh可以直接调用参数，不能用self.width、self.heigh作为变量获取，
        # 因为self.width、self.heigh 在模块中已经FigureCanvasQTAgg模块中使用，这里定义会造成覆盖
        # 我们生成的是正方形画布，所以建议长宽一致为width
        self.figs = Figure(figsize=(width, width), dpi=dpi)
        super(MyMplCanvas, self).__init__(self.figs)  # 在父类种激活self.fig， 否则不能显示图像（就是在画板上放置画布）
        self.axes = self.figs.add_subplot(111)  # 添加绘图区
        self.infoList = self.update_figure()
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def mat_plot_drow(self):
        pass

    def update_figure(self):
        pass


def get_twist_graphene(angle=0.0, size=20):
    bs = 100
    Super = size
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    thetaAngle = np.deg2rad(angle)
    xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
    s1 = np.stack((mox, moy), axis=-1)
    s2 = np.stack((xTwist, yTwist), axis=-1)
    out_S1, out_S2, out_S3, out_S4 = calDistance(s1, s2, r=r)
    content = []
    totalArea = calTotal(mox, bs=bs)
    ballNum = len(mox)
    overlapNum = len(out_S3)
    content.append(ballNum)
    content.append(overlapNum)
    content.append(totalArea)
    overLapArea = sumArea(out_S3, out_S4, bs=bs)
    overLapRatio = overLapArea / totalArea
    content.append(overLapArea)
    content.append('%.4f' % (overLapRatio * 100) + '%')
    # print()
    log.logger.info('%d' % content[0] + '%.4f' % (content[1] / 1000) + \
                    '    ' + '%.4f' % (content[2] / 1000) + '    ' + str(content[3]) + '\n')
    return out_S1, out_S2, out_S3, out_S4, content


class MyStaicMplCanvas(MyMplCanvas):
    def __init__(self, *args, **kwargs):
        log.logger.debug('MyStaicMplCanvas init')
        MyMplCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        log.logger.debug('compute_initial_figure')
        outS1, outS2, outS3, outS4, infoList = get_twist_graphene(angle=self.angle, size=self.Size)
        # self.model = pb.Model(two_graphene_monolayers(), pb.circle(radius=1.5), twist_layers(angle=angle))
        self.axes.cla()
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.scatter(outS1[:, 0], outS1[:, 1], 0.1, color='blue')
        self.axes.scatter(outS2[:, 0], outS2[:, 1], 0.1, color='green')
        self.axes.scatter(outS3[:, 0], outS3[:, 1], 1, marker='*', color='red')
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas

    def update_figure(self):
        log.logger.debug('update_figure %.3f %d' % (self.angle, self.Size))
        self.figs.clf()  # 清理画布，这里是clf()
        self.axes = self.figs.add_subplot(111)  # 清理画布后必须重新添加绘图区
        self.axes.patch.set_alpha(0.5)  # 设置ax区域背景颜色透明度
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        outS1, outS2, outS3, outS4, infoList = get_twist_graphene(angle=self.angle, size=self.Size)
        # self.model = pb.Model(two_graphene_monolayers(), pb.circle(radius=1.5), twist_layers(angle=angle))

        self.axes.scatter(outS1[:, 0], outS1[:, 1], 0.1, color='blue')
        self.axes.scatter(outS2[:, 0], outS2[:, 1], 0.1, color='green')
        self.axes.scatter(outS3[:, 0], outS3[:, 1], 0.5, marker='*', color='red')
        self.figs.canvas.draw()  # 这里注意是画布重绘，self.figs.canvas
        self.figs.canvas.flush_events()  # 画布刷新self.figs.canvas
        return infoList


class ShowTest:

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('main.ui')
        self.ui.centralwidget.setStyleSheet("background-color: rgb(	240,248,255); ")
        self.setIcon()
        self.ui.menubar.setStyleSheet("background-color: rgb(255,250,250);"
                                      "QPushButton:pressed{background-color:rgb(0, 206, 209)}")
        self.ui.lbl_tip1.setStyleSheet("font-color:black;font-family:黑体;")
        self.ui.info_wgt.setStyleSheet("background-color: rgb(253,245,230);")
        self.ui.edt_angle.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.ui.text_area1.setStyleSheet("background-color: rgb(250,250,210);")
        self.ui.btn_showInfo.setStyleSheet("QPushButton{background-color: rgb(60,179,113); "
                                           "color:white;font-family:黑体;font-size:14;border-radius:8;"
                                           "selection-color:rgb(	255,218,185);}"
                                           "QPushButton:pressed{background-color:rgb(0,206,209)}; ")
        self.ui.tol_wgt1.hide()
        self.ui.file_wgt.hide()
        self.ui.lbl_infoshow.setText('<font face="verdana" color="red">信息面板</font>')
        self.ui.act_run.triggered.connect(self.onShow)
        self.ui.btn_showInfo.clicked.connect(self.onShow)
        self.ui.act_exit.triggered.connect(self.onSignOut)
        self.ui.act_mail.triggered.connect(self.about)
        self.ui.act_tips.triggered.connect(self.tips)
        self.ui.act_version.triggered.connect(self.version)
        self.ui.act_param.triggered.connect(self.showParam)
        self.ui.act_savefile.triggered.connect(self.saveFile)
        self.ui.act_createfile.triggered.connect(self.createFile)
        self.ui.act_openfile.triggered.connect(self.openFile)
        self.ui.act_log.triggered.connect(self.openLog)
        self.ui.file_close_btn.clicked.connect(self.closeFileWidgit)
        # self.ui.fig_wgt.setCentralWidget(self.ui.fig_wgt)
        # self.ui.fig_wgt.statusBar().showMessage('matplotlib', 2000)

    def setIcon(self):
        appIcon = QIcon("icon.ico")
        self.ui.setWindowIcon(appIcon)

    def closeFileWidgit(self):
        strText = """<span style=" font-family:'SimSun'; font-size:12pt; color:#000000;">输入您需要的角度来展示
                </span></p>
                <p style=" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0;
                 text-indent:0px;"><span style=" font-family:'SimSun'; font-size:12pt; color:#000000;">或使用</span>
                 <span style=" font-family:'SimSun'; font-size:12pt; text-decoration: underline; color:#000000;"> 
                 </span><span style=" font-family:'SimSun'; font-size:12pt; font-weight:600; color:#000000;">工具
                 </span><span style=" font-family:'SimSun'; font-size:12pt; font-weight:600; font-style:italic; 
                 text-decoration: underline; color:#000000;"> </span><span style=" font-family:'SimSun'; 
                 font-size:12pt; color:#000000;">来设定扩胞参数(当前为Graphene模式)</span>"""
        self.ui.text_area1.setText(strText)
        self.ui.file_wgt.hide()

    def openLog(self):

        fp = os.getcwd() + '\\\\all.log'
        if os.path.isfile(fp):
            f = open(fp, 'r', encoding='UTF-8')
            with f:
                data = f.read()
                self.ui.file_close_btn.setStyleSheet("QPushButton{background-color: rgb(	60,179,113); "
                                                     "color:white;font-family:黑体;font-size:14;border-radius:10;}"
                                                     "QPushButton:pressed{background-color:rgb(0, 206, 209)}; ")
                self.ui.file_wgt.show()
                self.ui.text_area1.setText('已打开文件：\n' + fp)
                # self.ui.text_area2.setDocumentTitle(fileNames[0])
                self.ui.text_area2.setText(data)
        else:
            data = "日志文件错误！"
            self.ui.text_area2.setText(data)
            QMessageBox.warning(
                self.ui,
                "警告",
                "请勿删除*.exe目录下创建 all.log 日志文件\n如已删除请新建 all.log 文件，谢谢！"
            )


    def createFile(self):
        fname = QFileDialog.getSaveFileName(None, "%.3f_info_saved" % ShareInfo.agl, directory="./",
                                            filter="文本文件(*.txt);;All (*.*)")
        # filter="All (*.*)"写入文件首先获取文件路径
        if fname[0]:  # 如果获取的路径非空
            f = open(fname[0], "w")  # 以写入的方式打开文件
            with f:
                data = self.ui.text_area1.toPlainText()  # 获取textEdit的str
                f.write(data)
            f.close()

    def openFile(self):

        dialog = QFileDialog()
        # 设置文件过滤器，这里是任何文件，包括目录噢
        dialog.setFileMode(QFileDialog.AnyFile)
        # 设置显示文件的模式，这里是详细模式
        dialog.setViewMode(QFileDialog.Detail)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            f = open(fileNames[0], 'r', encoding="GBK")
            with f:
                data = f.read()
                self.ui.file_close_btn.setStyleSheet("QPushButton{background-color: rgb(	60,179,113); "
                                                     "color:white;font-family:黑体;font-size:14;border-radius:10;}"
                                                     "QPushButton:pressed{background-color:rgb(0, 206, 209)}; ")
                self.ui.file_wgt.show()
                self.ui.text_area1.setText('已打开文件：\n' + fileNames[0])
                # self.ui.text_area2.setDocumentTitle(fileNames[0])
                self.ui.text_area2.setText(data)

    def saveFile(self):
        fname = QFileDialog.getSaveFileName(None, "%.3f_twist_saved" % ShareInfo.agl, directory="./",
                                            filter="lammps(*.data);;All (*.*)")
        # filter="All (*.*)"写入文件首先获取文件路径
        if fname[0]:  # 如果获取的路径非空
            f = open(fname[0], "w")  # 以写入的方式打开文件
            with f:
                data = saveFile_cif(ShareInfo.agl)  # 获取textEdit的str
                f.write(data)
            f.close()

    def onSignOut(self):
        self.ui.close()

    def showParam(self):
        self.ui.tol_wgt1.setStyleSheet("background-color: rgb(	100,149,237); "
                                       "color:white;font-family:黑体;")
        self.ui.tol_wgt1_btn.setStyleSheet("QPushButton{background-color: rgb(	60,179,113); "
                                           "color:white;font-family:黑体;font-size:14;border-radius:10;}"
                                           "QPushButton:pressed{background-color:rgb(0, 206, 209)}; ")
        self.ui.tol_wgt1_size.setStyleSheet("background-color: rgb(240,248,255);color:black;  ")
        self.ui.tol_wgt1.show()
        self.ui.tol_wgt1_btn.clicked.connect(self.setParam)

    def setParam(self):
        input_size = self.ui.tol_wgt1_size.text().strip()

        try:
            input_size = int(input_size)
        except:
            QMessageBox.warning(
                self.ui,
                "警告",
                "参数解析不正确，非整型数字！"
            )
            return
        log.logger.debug('input cell size %dX%d' % (input_size, input_size))
        self.ui.text_area1.setText("扩胞参数已设置为：%d X %d" % (input_size, input_size) + '\n默认为 20 X 20')
        ShareInfo.mySize = int(self.ui.tol_wgt1_size.text().strip())
        self.ui.tol_wgt1.hide()

    def version(self):
        QMessageBox.information(self.ui, "Version",
                                """
        版本信息 1.0.0
        使用“关于-联系”获取更多信息 """
                                )

    def onShow(self):
        input_angle = self.ui.edt_angle.text().strip()

        # sessionid = requests.Session()
        try:
            ShareInfo.agl = float(input_angle)
        except:
            QMessageBox.warning(
                self.ui,
                "警告",
                "浮点数角度解析不正确，非浮点数！"
            )
            return
        log.logger.debug('input angle %.3f' % ShareInfo.agl)
        plt.title('angle = %.3f°' % ShareInfo.agl)
        ShowFigDialog(ShareInfo.agl)
        strText = "已展示 %.3f° 旋转图案" % ShareInfo.agl + \
                  "\n\n当前总原子数：%d" % (ShareInfo.sc.infoList[0]) + '对' + \
                  "\n\n重叠原子数：%d" % (ShareInfo.sc.infoList[1]) + '对' + \
                  "\n\n总面积：%.4f" % (ShareInfo.sc.infoList[2] / 1000) + \
                  "\n\n重叠面积：%.4f" % (ShareInfo.sc.infoList[3] / 1000) + \
                  "\n\n重叠度：%s" % (ShareInfo.sc.infoList[4])
        self.ui.text_area1.setText(strText)

    def about(self):
        QMessageBox.about(self.ui, "关于",
                          """
    e-mail: iamwxyoung@njtech.edu.cn
    e-mail: iamwjxyan@njtech.edu.cn
    Copyright 2021 Prof. Yan Jiaxu's group."""
                          )

    def tips(self):
        QMessageBox.about(self.ui, "建议",
                          """
        通过计算给出的特殊角度字典合集，格式为{key=degree:value=super cell length * 10000%}
        cellLength = {'6.01': 1354.862, '7.34': 1109.275, '9.43': 863.924, 
                    '10.42': 1354.862,'11.64': 1213.489, '13.17': 619.086, 
                    '15.18': 931.340, '16.43': 994.197,'17.9': 790.779, 
                    '21.79': 375.771, '24.43': 1162.551, '26.01': 1262.374,
                    '27.8': 512.090, '29.41': 1398.815}
        """
                          )


class ShowFigDialog:
    def __init__(self, angle=0.0):
        ShareInfo.dilg = QUiLoader().load('dilg.ui')
        I = QtWidgets.QVBoxLayout(ShareInfo.dilg.dilg_wgt)
        log.logger.debug('figSize: %d  %d' % (int(ShareInfo.mySize / 2), ShareInfo.mySize * 25))
        ShareInfo.sc = MyStaicMplCanvas(ShareInfo.dilg.dilg_wgt, width=int(ShareInfo.mySize / 2),
                                        dpi=ShareInfo.mySize * 10,
                                        angle=ShareInfo.agl)
        I.addWidget(ShareInfo.sc)
        ShareInfo.dilg.setWindowTitle('twist %.2f° show' % angle)
        ShareInfo.dilg.show()


if __name__ == '__main__':
    resultDict, cellLength = getDicts()
    log = Logger('all.log', level='debug')
    app = QApplication([])
    ShareInfo.showTest = ShowTest()
    ShareInfo.showTest.ui.show()
    app.exec_()
    # Logger('error.log', level='error').logger.error('error')
