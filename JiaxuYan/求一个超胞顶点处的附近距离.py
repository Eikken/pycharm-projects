#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   求一个超胞顶点处的附近距离.py    
@Time    :   2021/9/6 13:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import time
import numpy as np
import xlwt
import pandas as pd
from scipy.spatial import distance
from JiaxuYan.twistGrapheneTest import genGraphene, overFlowDrop, normXY, matrixTransformation, calEuclidean
import matplotlib.pyplot as plt


def calCellArea(s_1, s_2, cL, Agl):
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    index_S3, index_S4 = calEuclidean(s_1, s_2)
    tmpS1 = s_1[index_S3]
    tmpS2 = s_2[index_S4]
    dis3 = distance.cdist(tmpS1, [[0, 0]], 'euclidean').min(axis=1)
    dis4 = distance.cdist(tmpS2, [[0, 0]], 'euclidean').min(axis=1)
    coe = 1

    index_S1 = np.where(dis1 <= coe * cL[str(Agl)] + 284)
    index_S2 = np.where(dis2 <= coe * cL[str(Agl)] + 284)
    index_S5 = np.where(dis3 < coe * cL[str(Agl)] + 284)
    index_S6 = np.where(dis4 < coe * cL[str(Agl)] + 284)
    outS1, outS2, outS3, outS4 = s_1[index_S1[0]], s_2[index_S2[0]], tmpS1[index_S5], tmpS2[index_S6]
    # out 1 2 是所有原子坐标，3 4是重叠的SuperCell坐标
    dis5 = distance.cdist(s_1, [outS3[4]], 'euclidean')  # 13.17>>3,2  15.18>>4,3
    dis6 = distance.cdist(s_2, [outS4[3]], 'euclidean')
    index_S7 = np.where(dis5 <= 200)
    index_S8 = np.where(dis6 <= 200)
    outS7, outS8 = s_1[index_S7[0]], s_2[index_S8[0]]
    data = pd.DataFrame(distance.cdist(outS7, outS8, 'euclidean'))
    data.to_excel('data/%.2f定点距离.xls' % Agl)
    plt.figure(figsize=(12, 12), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(outS3[:, 0], outS3[:, 1], 140, marker='*', color='red')
    plt.scatter(outS7[:, 0], outS7[:, 1], color='green')
    plt.scatter(outS8[:, 0], outS8[:, 1], color='blue')
    # plt.savefig('png/bigCell/%.2f_14X.png' % Agl)
    plt.show()


if __name__ == '__main__':
    t1 = time.time()
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359, '29.41': 1398.815212957022}
    bs = 100
    Super = 70
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)  # 注意你删除的原子的方式
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    # angle = float(input('角度：'))
    angle = 15.18
    thetaAngle = np.deg2rad(angle)
    xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
    s1 = np.stack((mox, moy), axis=-1)
    s2 = np.stack((xTwist, yTwist), axis=-1)
    calCellArea(s1, s2, cellLength, angle)
    t2 = time.time()
    print('Finish, use time ', t2 - t1, 's')
