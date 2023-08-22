#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   ABstack_single_SuperCell.py    
@Time    :   2022/1/12 10:13  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   计算AB堆叠的单个超胞附近点的距离关系
'''

import copy

from scipy.spatial import cKDTree

from JiaxuYan.绘制AB_AA_twist重叠度对比 import getDict, dropMethod
import math
import numpy as np
import xlwt
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd


def genABStack3D(amplify=10):
    # Super [int] 扩胞系数
    # we decide to use a cube date to make our calculate be more simple
    layerGap = 1.42  # 1.41999299
    a246 = 0.24595  #: [nm] unit cell length
    a142 = 0.142  #: [nm] carbon-carbon distance
    a246 *= 10
    a = (a246, 0, 0)
    b = (a246 / 2, a246 / 2 * math.sqrt(3), 0)
    c = (0, 0, 1)
    # 扩胞矩阵
    super_x = amplify
    super_y = amplify
    super_z = 1

    transformtion = np.array([[super_x, 0, 0],
                              [0, super_y, 0],
                              [0, 0, super_z]])

    lattice = np.array([a, b, c])
    newLattice = np.dot(lattice, transformtion)
    Frac1 = 0.0
    Frac2 = 1 / float(3)
    Frac3 = 2 / float(3)
    allAtomsLayer1 = []
    allAtomsLayer2 = []
    index = 1
    for i in range(super_x):
        for j in range(super_y):
            newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.00]
            newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, layerGap]
            allAtomsLayer1.append(newC1)
            allAtomsLayer2.append(newC2)
            index += 1
            newC3 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.00]
            newC4 = [(Frac3 + i) / super_x, (Frac3 + j) / super_y, layerGap]
            allAtomsLayer1.append(newC3)
            allAtomsLayer2.append(newC4)
            index += 1

    newAtoms1 = np.dot(np.array(allAtomsLayer1), newLattice)
    newAtoms2 = np.dot(np.array(allAtomsLayer2), newLattice)
    return newAtoms1, newAtoms2


def getAB_DropData(*args, **kwargs):
    # args = set1, set2
    # 在这里传入两层原子数据，返回删除后未旋转的两层原子坐标数据
    meanList = [np.mean(args[1][:, 0]), np.mean(args[1][:, 1]), np.mean(args[0][:, 0]), np.mean(args[0][:, 1])]
    x_list1 = args[0][:, 0] - meanList[0]
    y_list1 = args[0][:, 1] - meanList[1]
    z_List1 = args[0][:, 2]

    x_list2 = args[1][:, 0] - meanList[2]
    y_list2 = args[1][:, 1] - meanList[3]
    z_List2 = args[1][:, 2]
    # min(meanList) / 2.0 作为去除无效原子的半径，使用dropCircle删除xyList中的无效原子。
    x_Drop1 = np.delete(x_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    y_Drop1 = np.delete(y_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    x_Drop2 = np.delete(x_list2, np.where(dropMethod(x_list2, y_list2) > min(meanList) / 2.0))
    y_Drop2 = np.delete(y_list2, np.where(dropMethod(x_list2, y_list2) > min(meanList) / 2.0))
    z_Drop1 = np.delete(z_List1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    z_Drop2 = np.delete(z_List2, np.where(dropMethod(x_list2, y_list2) > min(meanList) / 2.0))
    return [x_Drop1, y_Drop1, z_Drop1, x_Drop2, y_Drop2, z_Drop2]


# 计算两个集合中的欧几里得距离，返回距离很近的点的合集
def calCellEuclidean(*args, **kwargs):
    # s1 为列标，s2为行标，求s2内的点到s1中每个点最近的，就得取行最小值。
    zlj = 3.75694830724795  # zlj=最邻近距离
    clj = 6.50722534956336  # clj=次邻近距离=zlj * sqrt(3) 因为超胞是正六边形， 依次类推，第三临近就是2*zlj
    # 单个超胞计算距离
    dis3 = distance.cdist(args[0][:, :2], [[0, 0]], 'euclidean')
    dis4 = distance.cdist(args[1][:, :2], [[0, 0]], 'euclidean')
    index_S3 = np.where(dis3 < 2 * zlj + 0.07)[0]  # 有用没用的精确度达到0.01吧
    index_S4 = np.where(dis4 < 2 * zlj + 0.07)[0]

    dis1 = distance.cdist(args[0][index_S3][:, :2], args[1][index_S4][:, :2], 'euclidean').min(axis=1)
    dis2 = distance.cdist(args[0][index_S3][:, :2], args[1][index_S4][:, :2], 'euclidean').min(axis=0)
    index_S1 = np.where(dis1 < 0.01)  # 有用没用的精确度达到0.01吧
    index_S2 = np.where(dis2 < 0.01)
    # df = pd.DataFrame(distance.cdist(s1, s2, 'euclidean')) # 数据转Excel
    # df.to_excel('data/%.3f°distance.xlsx'%angle, index=True, header=True)
    return index_S1, index_S2, index_S3, index_S4


def draw_AB(*args, **kwargs):
    # AB双层画
    # args = layer1, twisted layer2, over lap layer 1, over lap layer 2
    plt.clf()
    plt.figure(figsize=(6, 6), edgecolor='black')
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(args[0][:, 0], args[0][:, 1], 15, color='blue')
    # plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 2, color='green')
    plt.scatter(args[1][:, 0], args[1][:, 1], 15, color='green')
    # rP = dropRedPoint(np.array(redPoint))
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(args[2][:, 0], args[2][:, 1], 30, marker='+', color='red')
    plt.scatter(args[3][:, 0], args[3][:, 1], 30, marker='+', color='red')
    # plt.scatter(rP[:, 0], rP[:, 1], 10, marker='*', color='red')
    plt.title('show %.3f°' % kwargs['agl'])
    print("saved figure %.3f°" % kwargs['agl'])
    plt.show()


def calCellDistance(*args, **kwargs):
    angle = kwargs['agl']
    G1 = kwargs['G1']
    G2 = kwargs['G2']
    xDrop1, yDrop1, zDrop1, xDrop2, yDrop2, zDrop2 = [val for val in getAB_DropData(G1, G2)]
    # 层1
    s1 = np.stack((xDrop1, yDrop1, zDrop1), axis=-1)
    theta = np.deg2rad(angle)
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # twist matrix
    twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
    x_Drop2 = twistXY[0, :]
    y_Drop2 = twistXY[1, :]
    z_Drop2 = zDrop2
    # 旋转的层2
    s2 = np.stack((x_Drop2, y_Drop2, z_Drop2), axis=-1)
    # 计算距离
    inDx1, inDx2, inDx3, inDx4 = calCellEuclidean(s1, s2)
    s_3 = s1[inDx3]
    s_4 = s2[inDx4]
    s_1 = s_3[inDx1]  # 重合位点S1
    s_2 = s_4[inDx2]  # 重合位点S2

    draw_AB(s_3, s_4, s_1, s_2, agl=angle)


if __name__ == '__main__':
    # 0.77 1.05 1.35 1.41 1.47 1.54 1.61 1.7 1.79 1.58 1.65 1.74
    # Super = 60  # 180
    angleList, resultDict = getDict()
    print(angleList, resultDict)
    # nG1, nG2 = genABStack3D(Super)  # 双层交叉重叠
    # # listEm = cal4P3LStructure(G1=nG1, G2=nG2, agl=0)
    # agl = [21.79]
    # for i in agl:
    #     calCellDistance(G1=nG1, G2=nG2, agl=i)
    a = 1
    b = a
    a = 2
    print(a == b)
    # plt.plot(np.array(pSet)[:, 0], np.array(pSet)[:, 2], marker='+', label='AA')
    # plt.plot(np.array(pSet)[:, 0], np.array(pSet)[:, 3], marker='*', label='AB')
    # plt.plot(np.array(pSet)[:, 0], np.array(pSet)[:, 4], marker='>', label="AB'")
    # # plt.plot(np.array(pSet)[:, 0], np.array(pSet)[:, 1], marker='+')
    # plt.title('tl != sl != hl')
    # plt.legend()
    # # plt.title('triangle length == square length')
    # plt.show()
    # df = pd.DataFrame(listEm, columns=['angle', 'length'])
    # df.to_excel('data/angle-length.xls')
print('finish')
