#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   twistSuperCell.py    
@Time    :   2021/5/27 17:28  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   旋转大范围的SuperCell表示成箭头的形式
'''

from JiaxuYan.twistGrapheneTest import *


def calSymPoint(point):
    # k1 = (point[0][1] - point[1][1]) / (point[0][0] - point[1][0])
    # k2 = -1 / k1  # K是斜率
    A = point[0][1] - point[1][1]  # y1-y2
    B = point[1][0] - point[0][0]  # x2-x1
    C = point[0][0] * point[1][1] - point[0][1] * point[1][0]  # x1*y2-y1*x2
    symmetryPoint(A, B, C, 0.0, 0.0)
    # [(point[0][0] + point[1][0]) / 2, (point[0][1] + point[1][1]) / 2],
    return symmetryPoint(A, B, C, 0.0, 0.0)


def symmetryPoint(A, B, C, x3, y3):
    """计算一般情况的直线对称点，根据斜率关系推导的数学关系式"""
    x = x3 - 2 * A * ((A * x3 + B * y3 + C) / (A * A + B * B))
    y = y3 - 2 * B * ((A * x3 + B * y3 + C) / (A * A + B * B))
    return [x, y]


def findCellCenter(s_1, s_2, a):
    index_S3, index_S4 = calEuclidean(s_1, s_2)
    tmpS1 = s_1[index_S3]
    tmpS2 = s_2[index_S4]
    coe = 1.0  # coe = 2.64575
    G3 = 3 ** 0.5
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    index_S1 = np.where(dis1 <= cellLength[str(a)] + 7)  # 加70作为胞半径
    index_S2 = np.where(dis2 <= cellLength[str(a)] + 7)
    # outS1, outS2, = s_1[index_S1[0]], s_2[index_S2[0]]
    dis3 = distance.cdist(tmpS1, [[0, 0]], 'euclidean').min(axis=1)
    dis4 = distance.cdist(tmpS2, [[0, 0]], 'euclidean').min(axis=1)
    index_S5 = np.where(dis3 < coe * cellLength[str(a)] + 7)
    index_S6 = np.where(dis4 < coe * cellLength[str(a)] + 7)
    index_S7 = np.where(dis3 < cellLength[str(a)] + 7)
    index_S8 = np.where(dis4 < cellLength[str(a)] + 7)
    outS3, outS4, outS5, outS6 = tmpS1[index_S5], tmpS2[index_S6], tmpS1[index_S7], tmpS2[index_S8]
    indexDict = {0: [0, 1], 1: [1, 3], 2: [3, 5], 3: [5, 4], 4: [4, 2], 5: [2, 0]}  # 边长系数矩阵仅适用于单个SuperCell
    SuperCellDict = {}  # { 0:[x1, y1], 1:[x2, y2]...}
    for i in range(6):
        pointSet = outS5[indexDict[i]]
        symPoint = calSymPoint(pointSet)
        SuperCellDict[i] = symPoint
    figSize = 10
    if cellLength[str(a)] > 900:
        figSize = 15
    plt.figure(figsize=(figSize, figSize), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(outS3[:, 0], outS3[:, 1], 60, marker='*', color='red')
    calAllDistance(s_1, s_2, cellLength, a)
    # outS1Dict, outS2Dict = {}, {}
    # drawArrow(s_1, s_2, outS1Dict, outS2Dict, a)
    for i in range(6):
        tmpDict1, tmpDict2 = {}, {}
        tmpDis1 = distance.cdist(s_1, [SuperCellDict[i]], 'euclidean')
        tmpDis2 = distance.cdist(s_2, [SuperCellDict[i]], 'euclidean')
        tmpIndex1 = np.where(tmpDis1 <= cellLength[str(a)] + 7)
        tmpIndex2 = np.where(tmpDis2 <= cellLength[str(a)] + 7)
        tmpOutS1, tmpOutS2 = s_1[tmpIndex1[0]], s_2[tmpIndex2[0]]
        cellDis1 = distance.cdist(tmpOutS1, [SuperCellDict[i]], 'euclidean')
        cellDis2 = distance.cdist(tmpOutS2, [SuperCellDict[i]], 'euclidean')
        # plt.scatter(tmpOutS1[:, 0], tmpOutS1[:, 1], color=randomcolor())
        # plt.scatter(tmpOutS2[:, 0], tmpOutS2[:, 1], color=randomcolor())
        for h, j in zip(cellDis1, cellDis2):
            hh = int(np.round(h))
            jj = int(np.round(j))
            if hh in tmpDict1:tmpDict1[hh] += 1
            else:tmpDict1[hh] = 1
            if jj in tmpDict2:tmpDict2[jj] += 1
            else:tmpDict2[jj] = 1
        for h in tmpDict1.keys():
            tmpDict1[h] = np.where(np.round(tmpDis1) == h)[0]
        for h in tmpDict2.keys():
            tmpDict2[h] = np.where(np.round(tmpDis2) == h)[0]
        tKset = tmpDict1.keys() ^ tmpDict2.keys()
        for tK in tKset:
            if tK in tmpDict2.keys():
                # print(i, 'in tmpDict2')
                if tK + 1 in tmpDict2:
                    tmpDict2[tK + 1] = sorted(np.append(tmpDict2[tK + 1], tmpDict2[tK]))
                elif tK-1 in tmpDict2.keys():
                    tmpDict2[tK - 1] = sorted(np.append(tmpDict2[tK - 1], tmpDict2[tK]))
                tmpDict2.pop(tK)
            # if tK in tmpDict1.keys():
            #     # print('in tmpDict1')
            #     if tK+1 in tmpDict1.keys():
            #         tmpDict1[tK + 1] = sorted(np.append(tmpDict1[tK + 1], tmpDict1[tK]))
            #     elif tK-1 in tmpDict1.keys():
            #         tmpDict1[tK - 1] = sorted(np.append(tmpDict1[tK + 1], tmpDict1[tK]))
            #     tmpDict1.pop(tK)
        # num = 1
        sortDict1 = dict(sorted(tmpDict1.items(), key=lambda x: x[0]))
        sortDict2 = dict(sorted(tmpDict2.items(), key=lambda x: x[0]))
        # for k1, k2 in zip(sortDict1, sortDict2):
        #     # X1 = s_1[tmpDict1[k1][0]]
        #     # X2 = s_2[tmpDict2[k2][0]]
        #     print(num, len(tmpDict1[k1]) == len((tmpDict2[k2])),
        #           k1, '>>', len(tmpDict1[k1]), '\t', k2, '>>', len(tmpDict2[k2]))
        #     num += 1
        drawArrow(s_1, s_2, tmpOutS1, sortDict1, sortDict2, a)
        # plt.scatter(tmpOutS1[:, 0], tmpOutS1[:, 1], color=randomcolor())
        # plt.scatter(tmpOutS2[:, 0], tmpOutS2[:, 1], color=randomcolor())
        # break
    # plt.savefig('png/bigArrow/Arrow_%.2f.png' % a)
    print('save fig %.2f' % a)


def draw():
    plt.scatter([0], [1], color=randomcolor())
    plt.show()


def calSpCelDistance(s_1, s_2, cD, a):
    coe = 2.64575
    G3 = 3 ** 0.5
    dis1 = distance.cdist(s_1, [[0, 0]], 'euclidean')
    dis2 = distance.cdist(s_2, [[0, 0]], 'euclidean')
    index_S1 = np.where(dis1 <= coe * cellLength[str(a)] + 7)  # 加70作为胞半径
    index_S2 = np.where(dis2 <= coe * cellLength[str(a)] + 7)
    outS1, outS2, = s_1[index_S1[0]], s_2[index_S2[0]]
    plt.figure(figsize=(8, 8), edgecolor='black')
    plt.xticks([])
    plt.yticks([])
    # plt.scatter(outS1[:, 0], outS1[:, 1], 30, color='blue')
    # plt.scatter(outS2[:, 0], outS2[:, 1], 30, color='green')
    # plt.scatter(outS4[:, 0], outS4[:, 1], 10, color='green')
    # plt.plot(outS3[:, 0], outS3[:, 1],linestyle='--', color='grey')


if __name__ == '__main__':
    t1 = time.time()
    bs = 100
    Super = 70
    xList, yList, zList, xMean, yMean = genGraphene(Super=Super, bs=bs)
    x_Drop, y_Drop = overFlowDrop(xList, yList, yMean)  # 注意你删除的原子的方式
    r = yMean
    mox = np.delete(x_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    moy = np.delete(y_Drop, np.where(normXY(x_Drop, y_Drop) > r))
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359}
    dd = {'7.34': 1109.275439,  '13.17': 619.0864237}
    # while True:
    # inputAngle = input('请输入逆时针旋转角度：')
    # if not inputAngle.replace(".", '').isdigit():
    #     break
    # angle = float(inputAngle)
    # x3, y3 = eval(input("\n请输入已知任意对称点X,Y坐标:"))
    print('start')
    for k in dd.keys():
        angle = float(k)
        thetaAngle = np.pi * angle / 180.0
        xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
        s1 = np.stack((mox, moy), axis=-1)
        s2 = np.stack((xTwist, yTwist), axis=-1)
        findCellCenter(s1, s2, angle)
        # break
        # calSpCelDistance(s1, s2, centerDict, angle)
    print('finish')