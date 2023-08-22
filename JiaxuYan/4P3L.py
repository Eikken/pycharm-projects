#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   4P3L.py    
@Time    :   2021/11/17 11:32  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   四点定三线结构
            list(itertools.combinations('ABC', 2))
            [('A', 'B'), ('A', 'C'), ('B', 'C')]
            邻位碳 ortho-carbon 简称o-carbon
            间位碳 meta-或m-carbon
            对位碳 para-或p-carbon
'''

import copy
import itertools
import sys

import networkx as nx
from scipy.spatial import cKDTree
from JiaxuYan.绘制AB_AA_twist重叠度对比 import getDict, dropMethod
import math
import numpy as np
import xlwt
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd

from JiaxuYan.绘制重叠和公式曲线 import getLineABC

c0 = 0.335  # 0.335
rad = 0.04925590198432363
AOB = 2.8221552998118007


def TriangleArea(a):
    return math.sqrt(3) * a ** 2 / 4.0


def SquareArea(a):
    return a ** 2


def HexagonArea(a):
    return math.sqrt(3) * 3 * a ** 2 / 2.0


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
def Euclidean(*args, **kwargs):
    # s1 为列标，s2为行标，求s2内的点到s1中每个点最近的，就得取行最小值。
    dis1 = distance.cdist(args[0][:, :2], args[1][:, :2], 'euclidean').min(axis=1)
    dis2 = distance.cdist(args[0][:, :2], args[1][:, :2], 'euclidean').min(axis=0)
    index_S1 = np.where(dis1 < 0.009)  # 有用没用的精确度达到0.01吧
    index_S2 = np.where(dis2 < 0.009)
    # df = pd.DataFrame(distance.cdist(s1, s2, 'euclidean')) # 数据转Excel
    # df.to_excel('data/%.3f°distance.xlsx'%angle, index=True, header=True)
    return index_S1, index_S2


def cal_3D_cosine(*args):
    # return the cos value of double vector
    # here we need a vector transform, OA = [1,1,0], OB = [2,2,2], AB=OB-OA
    # [x0,y0,z0] is which sample vertical Vector we need
    bV, tV = [val for val in args]
    x0, y0, z0 = [0, 0, 1]
    x1, y1, z1 = [bV[index] for index in range(3)]
    x2, y2, z2 = [tV[index] for index in range(3)]
    x2, y2, z2 = x2 - x1, y2 - y1, z2 - z1
    cos_b_t = (x0 * x2 + y0 * y2 + z0 * z2) / \
              (np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2))
    return cos_b_t


def cosineNearest(*args):
    # here we calculate six the nearest point to return for function calNearestStructure()
    layerG1, layerG2 = [i for i in args]
    vDown = []  # layer 1 vector
    vUp = []  # layer 2 vector
    tmpList = []
    for basisVector in layerG1:
        # point in layer1 [x, y, z]
        dis = distance.cdist(layerG2, [basisVector], metric='euclidean')
        pointNum = 1
        p_1_inDx = pd.Series(list(dis[:, 0])).sort_values().index[:pointNum]
        tmp6Point = layerG2[p_1_inDx]  # while layer1 to layer2 one nearest point in layer2
        for i in range(pointNum):
            tmpVector = tmp6Point[i]
            c3d = cal_3D_cosine(basisVector, tmpVector)
            # tmpList.append(c3d)
            if np.rad2deg(np.arccos(c3d)) <= minAgl:
                vDown.append(basisVector)
                vUp.append(tmpVector)  # vector down & up cosine little enough then bind them
    return np.array(vDown), np.array(vUp)


def calTriangleStructure(*args, **kwargs):
    layerG1 = kwargs['G1']
    layerG2 = kwargs['G2']
    bVst1, bVst2 = [i for i in args]  # 基础向量down, up
    for bV1, bV2 in zip(bVst1, bVst2):
        dis1 = distance.cdist(layerG1, [bV1], metric='euclidean')
        dis2 = distance.cdist(layerG2, [bV2], metric='euclidean')
        point1Num = 4
        point2Num = 4
        ly1_3_inDx = pd.Series(list(dis1[:, 0])).sort_values().index[:point1Num]
        ly2_3_inDx = pd.Series(list(dis2[:, 0])).sort_values().index[:point2Num]
        tmp3P1 = layerG1[ly1_3_inDx]
        tmp3P2 = layerG2[ly2_3_inDx]
        eL1 = list(itertools.combinations(tmp3P1, 2))
        eL2 = list(itertools.combinations(tmp3P2, 2))
        vectorNormalization(pointList1=np.array(eL1), pointList2=np.array(eL2))
        for i in range(3):
            # print(eL1[i], '\n', eL2[i], '\n')
            plt.plot(np.array(eL1[i])[:, 0], np.array(eL1[i])[:, 1], 5, color='red')
            plt.plot(np.array(eL2[i])[:, 0], np.array(eL2[i])[:, 1], 5, color='black')


def calLmdLength(*args):
    s1, s2 = [i for i in args]  # coincidence point of up and down, calculate lamda
    dis = distance.cdist(s1, s1, 'euclidean')
    mark = dis >= 10
    minDistance = distance.cdist(s1, s1, 'euclidean')[mark].min()
    # print(minDistance)
    return minDistance
    #
    # try:
    #     minDistance = distance.cdist(s1, s1, 'euclidean')[mark].min()
    #     minDistance2 = distance.cdist(s1, s1, 'euclidean')[dis > minDistance+1].min()
    # except:
    #     return 10
    # if (minDistance2 - minDistance) < 1.46:
    #     print(minDistance2, ' and ',  minDistance)
    #     return minDistance2
    # return minDistance


# [[  0.           1.41999299 105.7585     183.19560613 104.55044922,  105.76803252 183.89320373 210.298037   182.47324274 183.19560613]

def cal4P3LStructure(*args, **kwargs):
    # we need to draw 4 points and 3 lines structure
    # 根据每个单层构建图结构
    layerG1 = kwargs['G1']
    layerG2 = kwargs['G2']
    agl = kwargs['agl']
    emptyList = []

    xDrop1, yDrop1, zDrop1, xDrop2, yDrop2, zDrop2 = [val for val in getAB_DropData(layerG1, layerG2)]
    s1 = np.stack((xDrop1, yDrop1, zDrop1), axis=-1)
    samVal = 0
    for i in range(880, 900):
        agl = i / 1000.0

        theta = np.deg2rad(agl)
        Matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])  # twist matrix
        twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
        x_Drop2 = twistXY[0, :]
        y_Drop2 = twistXY[1, :]
        z_Drop2 = zDrop2
        s2 = np.stack((x_Drop2, y_Drop2, z_Drop2), axis=-1)
        # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
        inDx1, inDx2 = Euclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
        s_1 = s1[inDx1]
        s_2 = s2[inDx2]
        if len(s_1) == 4:
            La = calLmdLength(s_1, s_1)  # obtain currently angle lamda value
            samVal = La
            print(agl, ' >> ', La)
            emptyList.append([agl, La])
        else:
            print(agl)
            continue

        # pointDown, pointUp = cosineNearest(s1, s2)  # inDx3, inDx4 是余弦距离较为近的点 ndarray
        # # draw_AB(s1, s2, s_1, s_2, agl=agl, pD=pointDown, pU=pointUp)
        # # calTriangleStructure(pointDown, pointUp, G1=s1, G2=s2)
        # # for i in s_1:
        #     # calNearestStructure(i, G1=s1, G2=s2)
        # # # for i in range(3):
        # # #     # print(eL1[i], '\n', eL2[i], '\n')
        # # #     plt.plot(np.array(eL1[i])[:, 0], np.array(eL1[i])[:, 1], 5, color='red')
        # # #     plt.plot(np.array(eL2[i])[:, 0], np.array(eL2[i])[:, 1], 5, color='black')
    return emptyList


def vectorNormalization(*args, **kwargs):
    # 向量归一化
    pL1 = kwargs['pointList1']
    pL2 = kwargs['pointList2']
    for i in range(3):
        baseVector1 = pL1[i][1] - pL1[i][0]
        baseVector2 = pL2[i][1] - pL2[i][0]
        baseVector1 = baseVector1 / sum(np.abs(baseVector1))
        baseVector2 = baseVector2 / sum(np.abs(baseVector2))
        # print(baseVector1, baseVector2)
        # 一两个点就是一组向量


def calNearestStructure(*args, **kwargs):
    # x, y, z = [i for i in args[0]]
    tmpList = []
    layerG1 = kwargs['G1']
    layerG2 = kwargs['G2']
    basisVector1 = args[0]  # 基础向量down
    # basisVector2 = args[1]  # 基础向量up
    dis1 = distance.cdist(layerG1, [basisVector1], metric='euclidean')
    dis2 = distance.cdist(layerG2, [basisVector1], metric='euclidean')
    # layerG2[np.where(dis == dis.min())[0]]
    # pd.Series(list).sort_values(ascending=False)
    point1Num = 4
    point2Num = 4
    ly1_3_inDx = pd.Series(list(dis1[:, 0])).sort_values().index[:point1Num]
    # ly2_3_inDx = pd.Series(list(dis2[:, 0])).sort_values().index[:point2Num]
    ##########################################################################
    basisVector2 = layerG2[pd.Series(list(dis2[:, 0])).sort_values().index[0]]
    dis3 = distance.cdist(layerG2, [basisVector2], metric='euclidean')
    ly2_3_inDx = pd.Series(list(dis3[:, 0])).sort_values().index[:point2Num]
    # this one statement include affluent information
    tmp3P1 = layerG1[ly1_3_inDx]  # np.append([basisVector1], layerG1[ly1_3_inDx], axis=0)  # three nearest point in
    # layer1, add bV1 in
    tmp3P2 = layerG2[ly2_3_inDx]  # np.append([basisVector2], layerG2[ly2_3_inDx], axis=0)  # three nearest point in
    # layer2, add bV2 in
    # 转换成1对3的模式
    eL1 = list(itertools.combinations(tmp3P1, 2))
    eL2 = list(itertools.combinations(tmp3P2, 2))
    vectorNormalization(pointList1=np.array(eL1), pointList2=np.array(eL2))
    for i in range(3):
        plt.scatter(np.array(eL1[i])[:, 0], np.array(eL1[i])[:, 1], 5, color='red')
        plt.scatter(np.array(eL2[i])[:, 0], np.array(eL2[i])[:, 1], 5, color='red')
    # for i in range(3):
    #     # print(eL1[i], '\n', eL2[i], '\n')
    #     plt.plot(np.array(eL1[i])[:, 0], np.array(eL1[i])[:, 1], 5, color='red')
    #     plt.plot(np.array(eL2[i])[:, 0], np.array(eL2[i])[:, 1], 5, color='black')
    return tmpList


def normXY(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def dropRedPoint(rp):
    # 点的距离是142的点留下，其余删
    rrp = copy.deepcopy(rp)
    inDx = []
    for i in range(len(rp)):
        dis = distance.cdist(rp, [rp[i]], 'euclidean')  # .min(axis=1)
        mark = dis != 0.0
        minDistance = distance.cdist(rp, [rp[i]], 'euclidean')[mark].min()
        if minDistance < 2:  # 判断AB 或者AA区域附近的点
            inDx.append(False)
        else:
            inDx.append(True)
    return rrp[inDx]


def TwoPointOneLine(pS):
    # AX+BY+C=0
    X1, X2 = [pS[[i], 0] for i in range(2)]
    Y1, Y2 = [pS[[i], 1] for i in range(2)]
    A = Y2 - Y1
    B = X1 - X2
    C = X2 * Y1 - X1 * Y2
    k = - A[0] / B[0]
    c = - C[0] / B[0]
    return [k, c, X1, X2]


def draw_AB(*args, **kwargs):
    # AB双层画
    # args = layer1, twisted layer2, over lap layer 1, over lap layer 2
    # rP = dropRedPoint(np.array(redPoint))
    # pDown = kwargs['pD']
    # pUp = kwargs['pU']
    agl = kwargs['agl']
    plt.clf()
    plt.figure(figsize=(12, 12), edgecolor='black')
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(args[0][:, 0], args[0][:, 1], 10, color='blue')
    # plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 2, color='green')
    plt.scatter(args[1][:, 0], args[1][:, 1], 10, color='green')
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(args[2][:, 0], args[2][:, 1], 20, marker='+', color='red')
    k, c, start, end = [i for i in TwoPointOneLine(args[2][[0, 2]])]
    xL = np.arange(start - 1, end + 1, 1)
    plt.plot(xL, k * xL + c, 50, color='red')
    # plt.plot(args[2][5:7, 0], args[2][5:7, 1], 20, marker='+', color='yellow')
    # plt.scatter(args[3][:, 0], args[3][:, 1], 50, marker='+', color='red')
    # plt.scatter(pDown[:, 0], pDown[:, 1], 10, marker='*', color='red')
    # plt.scatter(pUp[:, 0], pUp[:, 1], 10, marker='*', color='red')
    # plt.scatter(rP[:, 0], rP[:, 1], 10, marker='*', color='red')
    plt.title('show %.2f°' % agl)
    # plt.savefig('png//pingaddfig//%.3f°_line%d.png' % (agl, Super), dpi=200)
    # # print("saved figure %.2f°" % agl)
    plt.show()


def getPLABC(pS):
    X1, X2 = [pS[[i], 0] for i in range(2)]
    Y1, Y2 = [pS[[i], 1] for i in range(2)]
    A = Y2 - Y1
    B = X1 - X2
    C = X2 * Y1 - X1 * Y2
    return [A[0], B[0], C[0], X1, X2, Y1, Y2]


def PLDistance(**kwargs):
    # Calculate the points where the distance from the point to the straight line is less than 1.42
    s1 = kwargs['s1']
    s2 = kwargs['s2']
    s_1 = kwargs['s_1']
    s_2 = kwargs['s_2']
    minDis = 2.84
    # Get the points in the square near the straight line
    try:
        ABC = getPLABC(s_1[[0, 2]])
    except:
        print(len(s_1), ' s_1[0, 2] is out of index bounds')
        sys.exit(0)
    A, B, C, x1, x2, y1, y2 = [i for i in ABC]
    ss1 = s1[s1[:, 0] > x1 - 1]
    ss1 = ss1[ss1[:, 0] < 1]
    ss1 = ss1[ss1[:, 1] < y1 + 1]
    ss1 = ss1[ss1[:, 1] > -1]
    ss2 = s2[s2[:, 0] > x1 - 1]
    ss2 = ss2[ss2[:, 0] < 1]
    ss2 = ss2[ss2[:, 1] < y1 + 1]
    ss2 = ss2[ss2[:, 1] > -1]
    tmpS1 = []
    tmpS2 = []
    for v1, v2 in zip(ss1, ss2):
        dis1 = (A * v1[0] + B * v1[1] + C) / math.sqrt(A ** 2 + B ** 2)
        dis2 = (A * v2[0] + B * v2[1] + C) / math.sqrt(A ** 2 + B ** 2)
        if dis1.__abs__() <= minDis:
            tmpS1.append(v1)
        if dis2.__abs__() <= minDis:
            tmpS2.append(v2)
    return np.array(tmpS1), np.array(tmpS2)
    # layer2index =


def getHexagonLamda(**kwargs):
    # The Law of Cosines
    dibian = 1.42 / 2
    angle = kwargs['angle']
    # x = dibian/(2*np.sin(angle/2))
    x = dibian / np.sin(angle)

    return x


def calSquareLineNeighbor(*args, **kwargs):
    # use two point to determine a straight line
    # use line to calculate nearby points' cosine angle
    # in this function we can get square length and triangle length
    La = args[0]
    layerG1 = kwargs['G1']
    layerG2 = kwargs['G2']
    agl = kwargs['agl']
    emptyList = []
    xDrop1, yDrop1, zDrop1, xDrop2, yDrop2, zDrop2 = [val for val in getAB_DropData(layerG1, layerG2)]
    s1 = np.stack((xDrop1, yDrop1, zDrop1), axis=-1)
    theta = np.deg2rad(agl)
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # twist matrix
    twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
    x_Drop2 = twistXY[0, :]
    y_Drop2 = twistXY[1, :]
    z_Drop2 = zDrop2
    s2 = np.stack((x_Drop2, y_Drop2, z_Drop2), axis=-1)
    # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
    # inDx1, inDx2 = Euclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
    # s_1 = s1[inDx1]
    # s_2 = s2[inDx2]
    # # if len(s_1) <= 1:
    # #     print(agl, '>> len(s_1) <= 1')
    # #     return [1, 1, 1, 1, 1, 1]
    # # La = calLmdLength(s_1, s_1)
    # lineS1, lineS2 = PLDistance(s1=s1, s2=s2, s_1=s_1, s_2=s_2)  # ndarray [[x,y,z],]
    # tmpList = []
    # vDown, vUp = [], []
    # # print(La)
    # for basisVector in lineS1:
    #     # point in layer1 [x, y, z]
    #     dis = distance.cdist(lineS2, [basisVector], metric='euclidean')
    #     pointNum = 1
    #     p_1_inDx = pd.Series(list(dis[:, 0])).sort_values().index[:pointNum]
    #     tmp6Point = lineS2[p_1_inDx]  # while layer1 to layer2 one nearest point in layer2
    #     tmpVector = tmp6Point[0]
    #     c3d = cal_3D_cosine(basisVector, tmpVector)
    #     # tmpList.append(c3d)
    #     # vDown.append(basisVector)
    #     # vUp.append(tmpVector)  # vector down & up cosine little enough then bind them
    #     tmpList.append(c3d)
    #     if 0 < np.rad2deg(np.arccos(c3d)) < 17.5 or np.rad2deg(np.arccos(c3d)) > 37.5:
    #         vDown.append(basisVector)
    #         vUp.append(tmpVector)  # vector down & up cosine little enough then bind them
    #     else:
    #         break
    #         # break  # 就要前几个值就退出
    # disDown = distance.cdist(np.array(vDown)[:, :2], np.array(vDown)[:, :2]).max()
    # squLen = La - disDown * 2
    # triLen = disDown * math.sqrt(3) * 2
    # hexLen = getHexagonLamda(angle=theta)
    # x = np.arange(len(tmpList))
    # print(np.rad2deg(np.arccos(tmpList)))
    # plt.scatter(x, np.rad2deg(np.arccos(tmpList)))

    plt.figure(figsize=(20, 20))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(s1[:, 0], s1[:, 1], 10, color='black')
    plt.scatter(s2[:, 0], s2[:, 1], 10, color='black')
    # plt.scatter(s_1[:, 0], s_1[:, 1], 50, marker='+', color='red')
    # # plt.scatter(np.array(vDown)[:, 0], np.array(vDown)[:, 1])
    # # plt.scatter(np.array(vUp)[:, 0], np.array(vUp)[:, 1], 50, marker='+')
    plt.savefig('png//pingaddfig//multi angle//%.3f°%d.png' % (agl, Super), dpi=300)
    return 1  # [agl, La, squLen, triLen, hexLen, disDown]

    # draw_AB(s1, s2, s_1, s_2, agl=agl)


def calPercent(param):
    angle, S_AA, S_AB, S_ABb, S_super_cell = [i for i in param]
    AA = S_AA / S_super_cell * 100
    AB = S_AB / S_super_cell * 100
    ABb = S_ABb / S_super_cell * 100
    allPercent = 100 - (AA + AB + ABb)
    dic = {
        'angle': '%.3f°' % angle,
        'AA': '%.6f%%' % AA,
        'AB': '%.6f%%' % AB,
        'ABb': '%.6f%%' % ABb,
        'GAP': '%.6f%%' % allPercent,
    }
    return [angle, AA, AB, ABb, allPercent], dic


def calGapList(*args):
    gl, La, sl, tl, hl, dD = [val for val in args[0]]
    l = (La - 9.838) / (1 + math.sqrt(3))
    hl = l * math.sqrt(3)
    sl = l
    S_super_cell = 1.5 * math.sqrt(3) * La ** 2
    S_AA = 1.5 * math.sqrt(3) * hl ** 2  # hexagon

    S_AB = math.sqrt(3) * hl ** 2 / 4.0
    S_AB = S_AB * 2  # triangle

    S_ABb = sl ** 2  # square
    S_ABb = S_ABb * 3

    S = [gl, S_AA, S_AB, S_ABb, S_super_cell]

    return calPercent(S)


def calLamList(*args):
    gl, La, sl, tl, hl, dD = [val for val in args[0]]
    S_super_cell = 1.5 * math.sqrt(3) * La ** 2
    S_AA = 1.5 * math.sqrt(3) * hl ** 2  # hexagon

    S_AB = math.sqrt(3) * tl ** 2 / 4.0
    S_AB = S_AB * 2  # triangle

    S_ABb = sl ** 2  # square
    S_ABb = S_ABb * 3

    S = [gl, S_AA, S_AB, S_ABb, S_super_cell]

    return calPercent(S)


if __name__ == '__main__':
    # 0.77 1.05 1.35 1.41 1.47 1.54 1.61 1.7 1.79 1.58 1.65 1.74
    W_radius = 1.41
    Mo_radius = 1.40
    S_radius = 1.0899
    MoS2L = 3.1558  # 传参设置MoS2的原胞size
    Super = 200  # 180
    GAP = 0.24595 * 40  # in our first observation, gap value close to quadruple m-carbon bond length
    WIDTH = 0.24595 * 20
    minAgl = 10  # Now, 3.49 is the best param26.56505117707799
    angleList, resultDict = getDict()
    nG1, nG2 = genABStack3D(Super)  # 双层交叉重叠
    data = pd.read_excel(r'data/angle_LBM5-15.xls', sheet_name='Sheet1')
    dataSet = np.array(data)[np.where(data['label']==True)]
    # listEm = cal4P3LStructure(G1=nG1, G2=nG2, agl=0)
    agl = [0.88, 0.90, 0.93, 0.96, 0.99, 1.02, 1.05, 1.08, 1.12, 1.16, 1.2, 1.24, 1.29, 1.35, 1.41]
    genAngle = [0.0, 1.08, 1.41,1.48, 4.41, 5.09, 6.01, 7.34, 9.43, 11.64, 13.17, 15.18,
                16.43, 17.9, 21.79, 24.43, 26.01, 27.8]
    # reLen = cal4P3LStructure(G1=nG1, G2=nG2, agl=0.0)
    pSet = []
    for i in agl:
        print(agl.index(i)+1)
        reL = calSquareLineNeighbor(1, G1=nG1, G2=nG2, agl=i)

    # for i in dataSet[29:]:
    #     agl = i[0]
    #     reL = calSquareLineNeighbor(i[1], G1=nG1, G2=nG2, agl=agl)
    #     percentList, percentDic = calLamList(reL)
    #     print(reL, '\n', percentDic)
    #     print()
    #     AA, AB, ABb = 1, 1, 1
    #     p = percentList[1] * AA + percentList[2] * AB + percentList[3] * ABb
    #     agl = percentList[0]
    #     pSet.append([agl, p, percentList[1], percentList[2], percentList[3]])

    # plt.rcParams['font.sans-serif'] = 'SimHei'
    # plt.rcParams['axes.unicode_minus'] = False
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
