#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   生成MoS2.py    
@Time    :   2021/8/19 9:49
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd


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


def generateAB2H(L, s=5):
    a = (L, 0, 0)
    b = (L / 2, L / 2 * math.sqrt(3), 0)
    c = (0, 0, 20)
    # 扩胞矩阵
    super_x = s
    super_y = s
    super_z = 1

    transformtion = np.array([[super_x, 0, 0],
                              [0, super_y, 0],
                              [0, 0, super_z]])

    lattice = np.array([a, b, c])
    newLattice = np.dot(lattice, transformtion)
    # S1 = [2 / float(3), 1 / float(3), 0.14482600]
    # S2 = [2 / float(3), 1 / float(3), 0.35517400]
    # Mo1 = [1 / float(3), 2 / float(3), 0.25]

    # file1 = open('data/mapAA.in', 'w')
    # file1.write('%d %d %d 2' % (super_x, super_y, super_z))  # 基矢
    # file1.write('#\\By Celeste Young\n')
    Frac1 = 1 / float(3)
    Frac2 = 2 / float(3)
    allAtomsLayer1 = []
    allAtomsLayer2 = []
    index = 1
    for i in range(super_x):
        for j in range(super_y):
            newS1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.14482600]
            newS3 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.64482600]
            allAtomsLayer1.append(newS1)
            allAtomsLayer2.append(newS3)
            # file1.write('%d %d %d 0 %d\n' % (i, j, 0, index))
            index += 1
            newS2 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.35517400]
            newS4 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.85517400]
            allAtomsLayer1.append(newS2)
            allAtomsLayer2.append(newS4)
            # file1.write('%d %d %d 1 %d\n' % (i, j, 1, index))
            index += 1
            newMo1 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.25000000]
            newMo2 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.75000000]
            allAtomsLayer1.append(newMo1)
            allAtomsLayer2.append(newMo2)
            # file1.write('%d %d %d 0 %d\n' % (i, j, 0, index))
            index += 1
    newAtoms1 = np.dot(np.array(allAtomsLayer1), newLattice)
    newAtoms2 = np.dot(np.array(allAtomsLayer2), newLattice)
    return newAtoms1, newAtoms2
    # file1.close()
    #
    # with open('data/MoS2AB.data', 'w') as writer:
    #     writer.write('MoS2 By Celeste\n\n')
    #     writer.write('%d atoms\n' % (len(allAtomsLayer1) + len(allAtomsLayer1)))
    #     writer.write('2 atom types\n\n')
    #     writer.write('%7.3f %7.3f xlo xhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f ylo yhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (newLattice[1][0], 0.0, 0.0))
    #     writer.write('  Masses\n\n')
    #     writer.write('1 32.06\n')
    #     writer.write('2 95.94\n')
    #     writer.write('Atoms\n\n')
    #     index = 1
    #     for i, j in zip(newAtoms1, newAtoms2):
    #         if i[2] == 5:
    #             writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, i[0], i[1], i[2]))
    #             index += 1
    #             writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, j[0], j[1], j[2]))
    #             index += 1
    #         else:
    #             writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, i[0], i[1], i[2]))
    #             index += 1
    #             writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, j[0], j[1], j[2]))
    #             index += 1
    # print('generated AB2H')


def generateAA3R(L, s=5):
    a = (L, 0, 0)
    b = (L / 2, L / 2 * math.sqrt(3), 0)
    c = (0, 0, 20)
    # 扩胞矩阵
    super_x = s
    super_y = s
    super_z = 1
    transformtion = np.array([[super_x, 0, 0],
                              [0, super_y, 0],
                              [0, 0, super_z]])

    lattice = np.array([a, b, c])
    newLattice = np.dot(lattice, transformtion)
    Frac1 = 0.0
    Frac2 = 0.33441
    Frac3 = 0.66883

    allAtomsLayer1 = []  # layer1 是WS2
    allAtomsLayer2 = []  # layer2 是MnS2
    index = 1
    for i in range(super_x):
        for j in range(super_y):
            newS1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.22801]
            newS3 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.55652]
            allAtomsLayer1.append(newS1)
            allAtomsLayer2.append(newS3)
            index += 1
            newS2 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.38652]
            newS4 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.71503]
            allAtomsLayer1.append(newS2)
            allAtomsLayer2.append(newS4)
            index += 1
            newMo1 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.30683]
            newW1 = [(Frac3 + i) / super_x, (Frac3 + j) / super_y, 0.63535]
            allAtomsLayer1.append(newMo1)
            allAtomsLayer2.append(newW1)
            index += 1
    newAtoms1 = np.dot(np.array(allAtomsLayer1), newLattice)
    newAtoms2 = np.dot(np.array(allAtomsLayer2), newLattice)
    return newAtoms1, newAtoms2
    # with open('data/MoWS4AA.data', 'w') as writer:
    #     writer.write('MoWS4 By Celeste\n\n')
    #     writer.write('%d atoms\n' % (len(allAtomsLayer1) + len(allAtomsLayer1)))
    #     writer.write('3 atom types\n\n')
    #     writer.write('%7.3f %7.3f xlo xhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f ylo yhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (newLattice[1][0], 0.0, 0.0))
    #     writer.write('  Masses\n\n')
    #     writer.write('1 32.06\n')  # S
    #     writer.write('2 95.94\n')  # Mo
    #     writer.write('3 183.85\n')  # W
    #     writer.write('Atoms\n\n')
    #     index = 1
    #     for i, j in zip(newAtoms1, newAtoms2):
    #         if int(i[2]) == 6:
    #             writer.write('%d 2 %7.3f %7.3f %7.3f\n' % (index, i[0], i[1], i[2]))
    #             index += 1
    #             writer.write('%d 3 %7.3f %7.3f %7.3f\n' % (index, j[0], j[1], j[2]))
    #             index += 1
    #         else:
    #             writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, i[0], i[1], i[2]))
    #             index += 1
    #             writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, j[0], j[1], j[2]))
    #             index += 1
    # print('generated AA3R')


def draw_1_0(set1, set2, agl):
    xList1 = set1[:, 0] - np.mean(set1[:, 0])
    yList1 = set1[:, 1] - np.mean(set1[:, 1])
    zList1 = set1[:, 2]
    theta = np.deg2rad(agl)
    layer1_Mo = (zList1 == 5.00000)
    Mo_xList1 = xList1[layer1_Mo]
    Mo_yList1 = yList1[layer1_Mo]
    xList2 = set2[:, 0] - np.mean(set2[:, 0])
    yList2 = set2[:, 1] - np.mean(set2[:, 1])
    zList2 = set2[:, 2]
    layer2_Mo = (zList2 == 15.00000)

    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # 旋转矩阵

    twistXY = np.dot(Matrix, np.array([xList2, yList2]))
    xList2 = twistXY[0, :]
    yList2 = twistXY[1, :]
    Mo_xList2 = xList2[layer2_Mo]
    Mo_yList2 = yList2[layer2_Mo]
    plt.figure(figsize=(5, 4), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(Mo_xList1, Mo_yList1, 5, color='blue')
    plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 1, color='green')
    plt.scatter(xList2[~layer2_Mo], yList2[~layer2_Mo], 1, color='green')
    plt.scatter(Mo_xList2, Mo_yList2, 5, color='lightgreen')
    plt.show()


if __name__ == '__main__':
    W_radius = 1.41
    Mo_radius = 1.40
    S_radius = 1.0899
    MoS2L = 3.1558  # 传参设置MoS2的原胞size
    Super = 10
    nAs1, nAs2 = generateAB2H(MoS2L, s=Super)
    mAs1, mAs2 = generateAA3R(MoS2L, s=Super)
    # print()
    draw_1_0(nAs1, nAs2, agl=13.17)

print('finish')
