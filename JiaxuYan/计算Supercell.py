#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   计算Supercell.py    
@Time    :   2021/4/28 12:48  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import time
import numpy as np
import xlwt
import pandas as pd
from scipy.spatial import distance

from JiaxuYan.twistGrapheneTest import genGraphene, overFlowDrop, normXY, matrixTransformation, calEuclidean, \
    drawOverLap
import matplotlib.pyplot as plt
import pandas as pd


def drawSuperCell(set1, theta):
    plt.figure(figsize=(10, 10), edgecolor='black')
    # plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(set1[:, 0], set1[:, 1], 5, marker='*', color='green')
    plt.scatter(0, 0, 10, marker='*', color='black')
    # plt.savefig('png/小角度/低谷_%.2f°_len=%.6f_.png' % (theta, bianchang), dpi=1000)
    print('saved 低谷_%.2f°_len=%.6f_.png' % (theta, bianchang))
    plt.show()


def calSuperCell(set1):
    # dis_list = set1[:2] # 失败
    dis1 = distance.cdist(set1, [(0, 0)], metric='euclidean')
    minDistance = dis1.min(axis=0)
    index = np.where(dis1 <= minDistance + 7)  # 为什么加7？考虑到原子半径
    super_cell = set1[index[0]]

    dis_set = distance.cdist(super_cell[1:], [super_cell[0]], metric='euclidean')  # 计算出第一个点到集合内各点的距离，
    # 去除自身距离0
    minCell = dis_set.min()
    lenCell = minCell / np.tan(np.pi / 6.0)  # a / c = tan(π/6); 2 * c = 2 * (a / tan(π/6))
    return minCell, lenCell


def drawAllAtoms(s_1, s_2, a ):
    pass


if __name__ == '__main__':
    t1 = time.time()
    Super = 100
    bs = 100
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359, '29.41': 1398.815212957022}
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    ###############################
    # book = xlwt.Workbook()
    # sheet = book.add_sheet('sheet1')
    # title = ['theta_angle', 'side_length', 'center_length']
    # row = 0  # 行
    # col = 0  # 列
    # for t in title:
    #     sheet.write(row, col, t)
    #     col += 1
    ###############################
    # dataSet = pd.read_excel('E:\桌面文件备份\\twist\峰角.xlsx', sheet_name='Sheet2')
    # # dataSetLength = len(dataSet['角度'])
    # for d in dataSet['低谷']:
    #     if pd.isna(d):
    #         break
    #     angle = np.deg2rad(d)
    #     xTwist, yTwist = matrixTransformation(mox, moy, angle)
    #     s1 = np.stack((mox, moy), axis=-1)
    #     s2 = np.stack((xTwist, yTwist), axis=-1)
    #     indexS1, indexS2 = calEuclidean(s1, s2)
    #     sortS1 = sorted(s1[indexS1], key=lambda s1_values: s1_values[0] + s1_values[1])
    #     sortS2 = sorted(s2[indexS2], key=lambda s2_values: s2_values[0] + s2_values[1])
    #     bianchang, baoxinju = calSuperCell(s1[indexS1])
    #     drawSuperCell(s1[indexS1], d)
    #     #####################################
    #     row += 1
    #     print(row, '/ 30', ' >> ', d)
    #     col = 0
    #     contents = [d, bianchang, baoxinju]
    #     for j in contents:
    #         sheet.write(row, col, j)
    #         col += 1
    #     ####################################
    # print('saved data/%d_角度_边长_胞心距2.xls' % Super)
    # book.save('data/%d_角度_边长_胞心距2.xls' % Super)

    # special_list = [6.01, 7.34, 9.43, 10.42, 11.64, 13.17, 15.18, 16.43, 17.90, 21.79, 24.43, 26.01, 27.80, 29.84]
    # 29.84 1398.815212957022
    # print('special_list共', len(special_list), '个angle')
    # book = xlwt.Workbook()
    # sheet = book.add_sheet('sheet1')
    # title = ['theta_angle', 'side_length', 'cell_length']
    # row = 0  # 行
    # col = 0  # 列
    # for t in title:
    #     sheet.write(row, col, t)
    #     col += 1
    # for i in special_list:
    #     row += 1
    #     col = 0
    #     contents = []
    #     thetaAngle = np.pi * i / 180.0
    #     xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
    #     s1 = np.stack((mox, moy), axis=-1)
    #     s2 = np.stack((xTwist, yTwist), axis=-1)
    #     indexS1, indexS2 = calEuclidean(s1, s2)
    #     sortS1 = sorted(s1[indexS1], key=lambda s1_values: s1_values[0] + s1_values[1])
    #     sortS2 = sorted(s2[indexS2], key=lambda s2_values: s2_values[0] + s2_values[1])
    #     bianchang, baoxinju = calSuperCell(s1[indexS1])
    #     contents.append(i)
    #     contents.append(bianchang)
    #     contents.append(baoxinju)
    #     for j in contents:
    #         print(j)  # sheet.write(row, col, j)
    #         col += 1
    #     print(row)
    # book.save('data/super=%d.xls' % Super)
    # print('saved data/super=%d.xls' % Super)
    while True:
        angle = float(input('input angle 绘图>> '))
        if angle == 0.0:
            break
        thetaAngle = np.deg2rad(float(angle))
        # xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        # s1 = np.stack((mox, moy), axis=-1)
        # s2 = np.stack((xTwist, yTwist), axis=-1)
        # indexS1, indexS2 = calEuclidean(s1, s2)
        # sortS1 = sorted(s1[indexS1], key=lambda s1_values: s1_values[0] + s1_values[1])
        # sortS2 = sorted(s2[indexS2], key=lambda s2_values: s2_values[0] + s2_values[1])
        # bianchang, baoxinju = calSuperCell(s1[indexS1])
        # print(bianchang)
    #     angle = float(input('input angle >> '))
    #     if angle == 0.0:
    #         break
    #     thetaAngle = np.pi * angle / 180.0
        xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        s1 = np.stack((mox, moy), axis=-1)
        s2 = np.stack((xTwist, yTwist), axis=-1)
        indexS1, indexS2 = calEuclidean(s1, s2)
        sortS1 = sorted(s1[indexS1], key=lambda s1_values: s1_values[0] + s1_values[1])
        sortS2 = sorted(s2[indexS2], key=lambda s2_values: s2_values[0] + s2_values[1])
        bianchang, baoxinju = calSuperCell(s1[indexS1])
        # print(bianchang)
        print(s1[indexS1])
        # calSuperCell(s1[indexS1])
        # drawAllAtoms(s1, s2, angle)
        # drawSuperCell(s2[indexS2], angle)
        # drawOverLap(s1[indexS1], s2[indexS2], angle)
            # drawSuperCell(s1[indexS1], angle)
        # plt.show()
    #  [6.4,6.61,6.84*,7.93,8.26, 8.61, 10.99,12.36, 13.10, 14.11,14.62,] 29.84
    t2 = time.time()
    print('Finish, use time ', t2 - t1, 's')
