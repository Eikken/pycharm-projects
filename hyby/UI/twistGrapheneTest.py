#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   twistGrapheneTest.py    
@Time    :   2021/3/20 11:46  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   先生成石墨烯结构，然后zip打包[xList,yList]，进行M(θ)角度的旋转并进行plt.show()
        min.(axis = )none：整个矩阵; 0：每列; 1：每行

'''

import math
import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import xlwt
import time
import pandas as pd
from scipy.spatial import cKDTree


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


def savePeakData():
    titleList = ['edgeLength', 'midPerpendicular', 'overLapArea', 'overLapRatio', 'layers', 'span', 'arrowDistance',
                 'atomsPair']
    for k in resultDict:
        resultDict[k].append(resultDict[k][0] * np.cos(np.pi / 6))
    for k in cellLength.keys():
        angle = float(k)
        thetaAngle = np.pi * angle / 180.0
        xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        s1 = np.stack((mox, moy), axis=-1)
        s2 = np.stack((xTwist, yTwist), axis=-1)
        calAllDistance(s1, s2, cellLength, angle)
    book = xlwt.Workbook()  # 创建Excel
    for k, v in resultDict.items():
        sheet = book.add_sheet(k)
        row = 0  # 行
        col = 0  # 列
        for t in titleList:
            sheet.write(row, col, t)
            col += 1
        row += 1
        col = 0
        for j in range(4):
            sheet.write(row, col, v[j])
            col += 1
        row -= 1  # 保持同一行
        for vv in v[4]:
            row += 1
            col = 4
            sheet.write(row, col, row)
            col += 1
            sheet.write(row, col, vv[0])
            col += 1
            sheet.write(row, col, vv[1][0])
            col += 1
            sheet.write(row, col, vv[1][1])
    book.save('data/peak_data_14.xls')


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


def calCellArea(mx, cL, BS=100):
    return 1


if __name__ == '__main__':
    t1 = time.time()
    bs = 100  # 不可修改，会引起整理比例失调
    Super = 20  # 扩胞大小可修改，X*X倍扩胞皆可
    print('2.46 : 1.42 : 0.07')
    print('size: %d X %d' % (Super, Super))
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    # 绘制圆
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)  # 注意你删除的原子的方式
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359, '29.41': 1398.815212957022}
    resultDict = {'6.01': [1354.862355], '7.34': [1109.275439], '9.43': [863.9236077], '10.42': [1354.8623546323813],
                  '11.64': [1213.4891841297963], '13.17': [619.0864237], '15.18': [931.3409687], '16.43': [994.1971635],
                  '17.9': [790.7793624], '21.79': [375.771207], '24.43': [1162.550644], '26.01': [1262.3739541039336],
                  '27.8': [512.0898359], '29.41': [1398.815212957022]}
    dataSet = np.array(pd.read_excel(r'E:\桌面文件备份\twist\data record 0705.xlsx', sheet_name='Sheet2'))
    Aglist = [21.79]
    while True:
        try:
            kk = input('请输入任意浮点数角度>>')
            k = float(kk)
        except:
            print('浮点数解析失败:', kk)
            break
        thetaAngle = np.deg2rad(k)
        xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        s1 = np.stack((mox, moy), axis=-1)
        s2 = np.stack((xTwist, yTwist), axis=-1)
        out_S1, out_S2, out_S3, out_S4 = calDistance(s1, s2, r=r)
        drawOverLap(out_S1, out_S2, out_S3, k)
        # 输入特殊角度是下一行的注释来查看超胞边长、重叠度等数据
        # calAllDistance(s1, s2, cellLength, k)
        content = []
        totalArea = calTotal(mox, bs=bs)
        content.append(totalArea)
        overLapArea = sumArea(out_S3, out_S4, bs=bs)
        overLapRatio = overLapArea / totalArea
        content.append(overLapArea)
        content.append('%.4f' % (overLapRatio * 100) + '%')
        print('总面积\t\t重叠面积\t\t重叠度\t')
        print('%.4f' % (content[0] / 1000), '\t', '%.4f' % (content[1] / 1000), '\t', content[2], '\t')
        plt.show()
