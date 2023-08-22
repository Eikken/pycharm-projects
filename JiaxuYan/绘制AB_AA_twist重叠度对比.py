#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   绘制AB_AA_twist重叠度对比.py
@Time    :   2021/10/20 14:51  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   material：石墨烯
'''

import math
import numpy as np
import xlwt
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd


class Circle:
    def __init__(self, x, y, R):
        self.x = float(x)
        self.y = float(y)
        self.r = float(R)

    def calArea(self):
        return np.pi * self.r ** 2


def calShadow(c_1, c_2):
    # c1 c2 是两个圆的对象
    d = ((c_1.x - c_2.x) ** 2 + (c_1.y - c_2.y) ** 2) ** 0.5
    if d == 0:
        return c_1.calArea()
    if d > 0.14:
        print('[', c_1.x, ',', c_1.y, '] 和 [', c_2.x, ',', c_2.y, ']不重叠')
        return 0.0
    else:
        ang1 = np.arccos((c_1.r ** 2 + d ** 2 - c_2.r ** 2) / 2.0 / c_1.r / d)
        ang2 = np.arccos((-c_1.r ** 2 + d ** 2 + c_2.r ** 2) / 2.0 / c_2.r / d)
        area = ang1 * c_1.r ** 2 + ang2 * c_2.r ** 2 - d * c_1.r * np.sin(ang1)
        return area


def calTotalArea(initBox):
    # initBox 是整个删除完原子的单层结构的集合
    circle = Circle(0, 0, 0.07)
    pointsNum = len(initBox)
    total_area = pointsNum * circle.calArea()
    return total_area


# 求dropList中圆相交的总面积
def sumArea(set1, set2):
    emptyList = []
    for s in set1:
        minDistance = distance.cdist([s], set2, 'euclidean').min(axis=1)
        indexTuple = np.where(distance.cdist(set2, [s], 'euclidean') == minDistance)
        # set2[indexTuple[0]][0]是准确的数据
        result = calShadow(Circle(s[0], s[1], 0.07),
                           Circle(set2[indexTuple[0]][0][0], set2[indexTuple[0]][0][1], 0.07))
        emptyList.append(result)
    return sum(emptyList)


# 计算两个集合中的欧几里得距离，返回距离很近的点的合集
def calEuclidean(s_1, s_2):
    # s1 为列标，s2为行标，求s2内的点到s1中每个点最近的，就得取行最小值。
    dis1 = distance.cdist(s_1, s_2, 'euclidean').min(axis=1)
    dis2 = distance.cdist(s_1, s_2, 'euclidean').min(axis=0)
    index_S1 = np.where(dis1 < 0.01)  # 有用没用的精确度达到0.01吧
    index_S2 = np.where(dis2 < 0.01)
    # df = pd.DataFrame(distance.cdist(s1, s2, 'euclidean')) # 数据转Excel
    # df.to_excel('data/%.3f°distance.xlsx'%angle, index=True, header=True)
    return index_S1, index_S2


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


def genABStackGraphene(Super=10):
    # Super [int] 扩胞系数
    a246 = 0.24595  #: [nm] unit cell length
    a142 = 0.142  #: [nm] carbon-carbon distance
    a246 *= 10
    a = (a246, 0, 0)
    b = (a246 / 2, a246 / 2 * math.sqrt(3), 0)
    c = (0, 0, 20)
    # 扩胞矩阵
    super_x = Super
    super_y = Super
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
            newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.25]
            newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.75]
            allAtomsLayer1.append(newC1)
            allAtomsLayer2.append(newC2)
            index += 1
            newC3 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.25]
            newC4 = [(Frac3 + i) / super_x, (Frac3 + j) / super_y, 0.75]
            allAtomsLayer1.append(newC3)
            allAtomsLayer2.append(newC4)
            index += 1

    newAtoms1 = np.dot(np.array(allAtomsLayer1), newLattice)
    newAtoms2 = np.dot(np.array(allAtomsLayer2), newLattice)
    # with open('data/GraAB.data', 'w') as writer:
    #     writer.write('Graphene By Celeste\n\n')
    #     writer.write('%d atoms\n' % (len(allAtomsLayer1) + len(allAtomsLayer1)))
    #     writer.write('1 atom types\n\n')
    #     writer.write('%7.3f %7.3f xlo xhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f ylo yhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, newLattice[0][0]))
    #     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (newLattice[1][0], 0.0, 0.0))
    #     writer.write('  Masses\n\n')
    #     writer.write('1 12.011\n')
    #     writer.write('Atoms\n\n')
    #     index = 1
    #     for i, j in zip(newAtoms1, newAtoms2):
    #         writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, i[0], i[1], i[2]))
    #         index += 1
    #         writer.write('%d 1 %7.3f %7.3f %7.3f\n' % (index, j[0], j[1], j[2]))
    #         index += 1
    # print('generated AB Graphene file')
    return newAtoms1, newAtoms2


def genAAStackGraphene(Super=10):  # 返回新的坐标的大胞
    # Super [int] 扩胞系数
    a246 = 0.24595  #: [nm] unit cell length
    a142 = 0.142  #: [nm] carbon-carbon distance
    a246 *= 10
    a = (a246, 0, 0)
    b = (a246 / 2, a246 / 2 * math.sqrt(3), 0)
    c = (0, 0, 10)
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
    return newAllAtoms


def dropMethod(xx, yy):
    return (xx ** 2 + yy ** 2) ** 0.5


def getAB_DropData(set_1, set_2):
    # 在这里传入两层原子数据，我们直接处理好两层数据，并且把其中一层旋转，
    # 返回删除后未旋转的两层原子坐标数据
    meanList = [np.mean(set_2[:, 0]), np.mean(set_2[:, 1]), np.mean(set_1[:, 0]), np.mean(set_1[:, 1])]
    x_list1 = set_1[:, 0] - meanList[0]
    y_list1 = set_1[:, 1] - meanList[1]
    zList1 = set_1[:, 2]

    x_list2 = set_2[:, 0] - meanList[2]
    y_list2 = set_2[:, 1] - meanList[3]
    zList2 = set_2[:, 2]
    # min(meanList) / 2.0 作为去除无效原子的半径，使用dropCircle删除xyList中的无效原子。
    x_Drop1 = np.delete(x_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    y_Drop1 = np.delete(y_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    x_Drop2 = np.delete(x_list2, np.where(dropMethod(x_list2, y_list2) > min(meanList) / 2.0))
    y_Drop2 = np.delete(y_list2, np.where(dropMethod(x_list2, y_list2) > min(meanList) / 2.0))
    return [x_Drop1, y_Drop1, x_Drop2, y_Drop2]


def getAA_DropData(set_1):
    # 在这里传入两层原子数据，我们直接处理好两层数据，并且把其中一层旋转，
    # 返回删除后未旋转的两层原子坐标数据
    meanList = [np.mean(set_1[:, 0]), np.mean(set_1[:, 1])]
    x_list1 = set_1[:, 0] - meanList[0]
    y_list1 = set_1[:, 1] - meanList[1]
    zList1 = set_1[:, 2]
    # min(meanList) / 2.0 作为去除无效原子的半径，使用dropCircle删除xyList中的无效原子。
    x_Drop1 = np.delete(x_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    y_Drop1 = np.delete(y_list1, np.where(dropMethod(x_list1, y_list1) > min(meanList) / 2.0))
    return [x_Drop1, y_Drop1]


def draw_AB_0(set1, set2, agl):   # AA双层画
    dtList = getAB_DropData(set1, set2)
    xDrop1, yDrop1, xDrop2, yDrop2 = dtList[0], dtList[1], dtList[2], dtList[3]
    theta = np.deg2rad(agl)
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])  # 旋转矩阵
    twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
    xDrop2 = twistXY[0, :]
    yDrop2 = twistXY[1, :]
    s1 = np.stack((xDrop1, yDrop1), axis=-1)
    s2 = np.stack((xDrop2, yDrop2), axis=-1)
    # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
    inDx1, inDx2 = calEuclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
    s_1 = s1[inDx1]
    s_2 = s2[inDx2]
    plt.figure(figsize=(10, 10), edgecolor='black')
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(s1[:, 0], s1[:, 1], 15, color='red')
    # plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 2, color='green')
    plt.scatter(s2[:, 0], s2[:, 1], 5, color='blue')
    # plt.savefig('png/AA_stack_graphene_png/%.2f°_AA_twist.png' % agl)
    # print("saved figure %.2f°" % agl)
    plt.show()


def draw_AA_0(set_1, agl=None):  #  # AB画单层G
    dtList = getAA_DropData(set_1)
    xDrop1, yDrop1 = dtList[0], dtList[1]
    # xDrop1, yDrop1 是已经删除了无效原子的单层x, y 的坐标，
    # 千万不要动xDrop1, yDrop1他们本身！
    # 千万不要动xDrop1, yDrop1他们本身！
    # 千万不要动xDrop1, yDrop1他们本身！
    plt.close()
    plt.figure(figsize=(10, 10), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(xDrop1,  yDrop1, color='red')
    # plt.scatter(set_2[:, 0], set_2[:, 1], 5, color='blue')
    # plt.scatter(set_2[:, 0], set_2[:, 1], 5, color='blue')
    # plt.savefig('png/AA_stack_graphene_png/%.2f°_AA_twist.png' % agl)
    # print("saved figure %.2f°" % agl)
    plt.title("%.2f figure show"%agl)
    plt.show()


def draw_AB_1(set_1, set_2, agl):  # AA双层过渡画
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(set_1[:, 0], set_1[:, 1], 15, color='red')
    # plt.scatter(xList1[~layer1_Mo], yList1[~layer1_Mo], 2, color='green')
    plt.scatter(set_2[:, 0], set_2[:, 1], 5, color='blue')
    # plt.savefig('png/AA_stack_graphene_png/%.2f°_AA_twist.png' % agl)
    # print("saved figure %.2f°" % agl)
    plt.title("%.2f figure show" % agl)


def draw_AA_1(set_1, set_2, agl):  # AB双层过渡画
    plt.close()
    plt.figure(figsize=(6, 6), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter([0], [0], 50, marker='*', color='black')
    plt.scatter(set_1[:, 0], set_1[:, 1], 15, color='red')
    plt.scatter(set_2[:, 0], set_2[:, 1], 5, color='blue')
    # plt.savefig('png/AB_stack_graphene_png/%.2f°_AB_twist.png' % agl)
    # print("saved figure %.2f°" % agl)
    plt.title("%.2f figure show" % agl)
    plt.show()


def saveAB_StepData(set1, set2):
    dtList = getAB_DropData(set1, set2)
    xDrop1, yDrop1, xDrop2, yDrop2 = dtList[0], dtList[1], dtList[2], dtList[3]
    titleList = ['angle', 'overLapArea', 'atomsPair', 'overLapRatio']
    s1 = np.stack((xDrop1, yDrop1), axis=-1)
    allArea = calTotalArea(s1)
    book = xlwt.Workbook()  # 创建Excel
    sheet = book.add_sheet('Sheet1')
    row = 0  # 行
    col = 0  # 列
    for t in titleList:
        sheet.write(row, col, t)
        col += 1
    for i in range(6100):
        angle = i / 100.0
        row += 1  # 行加一
        print(row)
        col = 0  # 从第0列开始写
        content = []  # 临时内容列表写入excel文件
        # angle = float(input('输入角度>>'))
        theta = np.deg2rad(angle)
        Matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])  # 旋转矩阵
        twistXY = np.dot(Matrix, np.array([xDrop2, yDrop2]))
        x_Drop2 = twistXY[0, :]  # 旋转后的原子坐标集合
        y_Drop2 = twistXY[1, :]
        s2 = np.stack((x_Drop2, y_Drop2), axis=-1)
        # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
        inDx1, inDx2 = calEuclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
        s_1 = s1[inDx1]
        s_2 = s2[inDx2]
        overLapaArea = sumArea(s_1, s_2)
        # print(s_1,len(s_1),)
        overRatio = overLapaArea / allArea * 100
        content.append(angle)
        content.append(overLapaArea)
        content.append(len(s_1))
        content.append(overRatio)
        for j in content:
            sheet.write(row, col, j)
            col += 1
    book.save('data/%dX%d_AB_overLapRatio.xls' % (Super, Super))
    # drawS1_2(s1, s2, agl)


def saveAA_StepData(set_1):
    dtList = getAA_DropData(set_1)
    xDrop1, yDrop1 = dtList[0], dtList[1]  # 暂且认为1、2两层原子是完全重叠的
    titleList = ['angle', 'overLapArea', 'atomsPair', 'overLapRatio']
    s1 = np.stack((xDrop1, yDrop1), axis=-1)
    allArea = calTotalArea(s1)
    book = xlwt.Workbook()  # 创建Excel
    sheet = book.add_sheet('Sheet1')
    row = 0  # 行
    col = 0  # 列
    for t in titleList:
        sheet.write(row, col, t)
        col += 1
    for i in range(6100):
        # angle = float(input('输入角度>>'))
        row += 1  # 行加一
        print(row)
        col = 0  # 从第0列开始写
        content = []  # 临时内容列表写入excel文件
        angle = i/100.0
        theta = np.deg2rad(angle)
        Matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])  # 旋转矩阵
        twistXY = np.dot(Matrix, np.array([xDrop1, yDrop1]))
        xDrop2 = twistXY[0, :]  # 旋转后的原子坐标集合
        yDrop2 = twistXY[1, :]
        s2 = np.stack((xDrop2, yDrop2), axis=-1)  # xDrop2是drop1旋转后的点阵合集
        # 现在drop中存的都是基础原子坐标数据，下一步进行距离判定获取距离集合
        inDx1, inDx2 = calEuclidean(s1, s2)  # inDx1, inDx2 是距离极其近的点的下标合集，接近重合点
        s_1 = s1[inDx1]
        s_2 = s2[inDx2]
        # draw_AB_1(s_1,s_2,angle)
        # draw_AB_0(s1, s2, angle)
        overLapaArea = sumArea(s_1, s_2)
        # print(s_1,len(s_1),)
        overRatio = overLapaArea / allArea * 100
        content.append(angle)
        content.append(overLapaArea)
        content.append(len(s_1))
        content.append(overRatio)
        for j in content:
            sheet.write(row, col, j)
            col += 1
    book.save('data/%dX%d_AA_60°_Ratio.xls' % (Super, Super))


def getDict():
    # 好多信号不低的角度
    agl = [0.41, 0.692, 0.787, 0.77, 0.803, 1.05, 1.12, 1.16, 1.35, 1.41, 1.47, 1.54, 1.61, 1.7, 1.79, 1.58, 1.65,
           1.74, 2.88, 3.15, 3.48, 3.89, 4.41, 5.09, 6.01, 6.61, 7.34, 8.26, 8.61,
           9.43, 10.42, 10.99, 11.64, 11.99, 13.17, 14.31, 14.45,  14.62,
           15.18, 15.66, 16.43, 17.28, 17.9, 18.73, 19.03, 19.28, 19.65,
           19.93, 20.32, 20.67, 21.79, 23.04, 23.49, 23.71, 23.85, 24.02,
           24.43, 25.04, 25.46, 26.01, 26.75, 27.05, 27.8, 28.78, 29.41]

    rD = {'6.01': [1354.862], '7.34': [1109.275], '9.43': [863.924],
          '10.42': [1354.862], '11.64': [1213.489], '13.17': [619.086],
          '15.18': [931.340], '16.43': [994.197], '17.9': [790.779],
          '21.79': [375.771], '24.43': [1162.551], '26.01': [1262.374],
          '27.8': [512.090], '29.41': [1398.815]}
    return agl, rD


if __name__ == '__main__':
    W_radius = 1.41
    Mo_radius = 1.40
    S_radius = 1.0899
    MoS2L = 3.1558  # 传参设置MoS2的原胞size
    Super = 70
    # angleList, resultDict = getDict()
    nG1, nG2 = genABStackGraphene(Super)  # 双层交叉重叠
    # draw_AB_0(nG1, nG2, 6.01)
    # dG1 = genAAStackGraphene(Super)  # 双层层完全重叠
    # draw_AA_0(dG1, agl=0.0)
    # saveAA_StepData(dG1)
    # saveAB_StepData(nG1, nG2)
    # for i in range(10):
    #     angle = i / 100.0
    # draw_1_0(nG1, nG2, agl=9.43)

# print('finish')
