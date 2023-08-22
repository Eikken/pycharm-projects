#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   计算AA AB的比率.py    
@Time    :   2021/11/3 11:41  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   graphene 中六边形个数n>2时，顶点数为3n+4
             在小角度：∠AOB=2.8221552998118007，rad(AOB)=0.04925590198432363
             面积计算出的比率更大更直观，按原子比较不太合理。
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

c0 = 0.335  # 0.335
rad = 0.04925590198432363
AOB = 2.8221552998118007


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
    index_S1 = np.where(dis1 < 0.03)  # 有用没用的精确度达到0.01吧
    index_S2 = np.where(dis2 < 0.03)
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


def calPointCircleArea(*args, **kwargs):
    # x, y, z = [i for i in args[0]]
    tmpList = []
    layerG2 = kwargs['G2']
    basisVector = args[0]  # 基础向量
    dis = distance.cdist(layerG2, [basisVector], metric='euclidean')
    # layerG2[np.where(dis == dis.min())[0]]
    # 对list数组元素进行排序（默认从小到大）
    # from the observation we can see that only three to nine point around bottom layer atom
    # because others are not in keeping with AA or AB stack
    # 选取list数组元素中最大（最小）的n个值的索引
    # pd.Series(list).sort_values(ascending=False)  # 从大到小用不太到
    pointNum = 9
    p_3_9_inDx = pd.Series(list(dis[:, 0])).sort_values().index[:pointNum]
    # this one statement include affluent information
    tmp9Point = layerG2[p_3_9_inDx]  # six point in layer2
    # ## now six valid point of 3D cube top are all in "layerG2[p_3_6_inDx]"
    # here, we will calculate cosine value of double vector
    for i in range(pointNum):
        tmpVector = tmp9Point[i]
        c3d = cal_3D_cosine(basisVector, tmpVector)
        tmpList.append(c3d)
        if np.rad2deg(np.arccos(c3d).min()) <= minAgl:
            redPoint.append(tmpVector)
    return tmpList


def calVector(*args, **kwargs):
    # Transmit into layer1 and twist layer2
    # Calculate every point generate_circle(R=C-C keyLength) in layer1 to generate a Vector
    # towards to layer2, whether the cosine value is
    thisList = []
    layerG1 = kwargs['G1']
    layerG2 = kwargs['G2']
    agl = kwargs['agl']
    xDrop1, yDrop1, zDrop1, xDrop2, yDrop2, zDrop2 = [val for val in getAB_DropData(layerG1, layerG2)]
    ballNum = len(xDrop1)
    theta = np.deg2rad(agl)
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # 旋转矩阵
    twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
    xDrop2 = twistXY[0, :]
    yDrop2 = twistXY[1, :]
    s1 = np.stack((xDrop1, yDrop1, zDrop1), axis=-1)
    s2 = np.stack((xDrop2, yDrop2, zDrop2), axis=-1)
    # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
    inDx1, inDx2 = Euclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
    print('overlap >> ', len(inDx1[0]))
    s_1 = s1[inDx1]
    s_2 = s2[inDx2]
    # for Pin1 in s1:
    #     # point in layer1 [x, y, z]
    #     thisList.append(calPointCircleArea(Pin1, G2=s2))

    draw_AB(s1, s2, s_1, s_2, agl=kwargs['agl'])
    return thisList, ballNum


def dis_2(dis):
    return [sorted(i)[1] for i in dis]


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


def draw_AB(*args, **kwargs):
    # AB双层画
    # args = layer1, twisted layer2, over lap layer 1, over lap layer 2
    plt.clf()
    plt.figure(figsize=(7, 7), edgecolor='black')
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(args[0][:, 0], args[0][:, 1], 10, color='blue')
    # plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 2, color='green')
    plt.scatter(args[1][:, 0], args[1][:, 1], 10, color='green')
    # rP = dropRedPoint(np.array(redPoint))
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(args[2][:, 0], args[2][:, 1], 10, marker='+', color='red')
    plt.scatter(args[3][:, 0], args[3][:, 1], 10, marker='+', color='red')
    # plt.scatter(rP[:, 0], rP[:, 1], 10, marker='*', color='red')
    plt.title('show %.3f°' % kwargs['agl'])
    plt.savefig('png//pingaddfig//大角度//%.3f°_AB%dsample.png' % (kwargs['agl'], Super), dpi=200)
    print("saved figure %.3f°" % kwargs['agl'])
    plt.show()


if __name__ == '__main__':
    # 0.77 1.05 1.35 1.41 1.47 1.54 1.61 1.7 1.79 1.58 1.65 1.74
    W_radius = 1.41
    Mo_radius = 1.40
    S_radius = 1.0899
    MoS2L = 3.1558  # 传参设置MoS2的原胞size
    Super = 40

    minAgl = 3.49  # Now,  is the best param 26.56505117707799
    angleList, resultDict = getDict()
    nG1, nG2 = genABStack3D(Super)  # 双层交叉重叠
    # cosList, length = calVector(G1=nG1, G2=nG2, agl=13.17)
    dic = {'1.0': 0, '12.9': 0}
    redPoint = []
    # cosList, length = calVector(G1=nG1, G2=nG2, agl=6.01)
    count = 1
    # for i in angleList[:12]:
    c, v = calVector(G1=nG1, G2=nG2, agl=21.79)
        # print(count, '/ 12')
        # count+=1
    # for i in cosList:
    #     # print(np.rad2deg(np.arccos(i)).min())
    #     if np.rad2deg(np.arccos(i)).min() <= minAgl:  # np.rad2deg(np.arccos(i)).min() < 5.0:
    #         print(count, '>>', np.rad2deg(np.arccos(i)))
    #         count += 1
    # # #     elif 12.63 < np.rad2deg(np.arccos(i)).min() < 13.17:
    # # #         dic['12.9'] += 1
    # # # print(dic, end='  ')
    # # print(np.rad2deg())
    # print(length)
    # print(dic['1.0']/length, dic['12.9']/length)
    # dis = distance.cdist(nG1, nG1, 'euclidean')
    # print(dis)

    # dG1 = genAAStackGraphene(Super)  # 双层层完全重叠
    # draw_AA_0(dG1, agl=0.0)
    # saveAA_StepData(dG1)
    # saveAB_StepData(nG1, nG2)
    # for i in range(10):
    #     angle = i / 100.0
    # draw_1_0(nG1, nG2, agl=9.43)

print('finish')
