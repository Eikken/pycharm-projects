#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   求百分比.py
@Time    :   2021/11/28 16:23
@E-mail  :   iamwxyoung@qq.com
@Tips    :
'''
import math
import numpy as np
import matplotlib.pyplot as plt


def getFourArea(**kwargs):
    l = kwargs['l']
    L = kwargs['L']
    agl = kwargs['agl']
    hexagon_length = l * math.sqrt(3)
    square_length = hexagon_length
    triangle_length = l / 2.0
    S_super_cell = 1.5 * math.sqrt(3) * L ** 2
    S_AA = 1.5 * math.sqrt(3) * hexagon_length ** 2
    S_AB = math.sqrt(3) * hexagon_length ** 2 / 4.0
    # S_AB = 0.5 * math.sqrt(3) * triangle_length ** 2
    S_AB = S_AB * 2
    S_ABb = square_length ** 2
    S_ABb = S_ABb * 3
    return [agl, S_AA, S_AB, S_ABb, S_super_cell]


def calPercent(param):
    agl, S_AA, S_AB, S_ABb, S_super_cell = [i for i in param]
    AA = S_AA / S_super_cell * 100
    AB = S_AB / S_super_cell * 100
    ABb = S_ABb / S_super_cell * 100
    allPercent = 100 - (AA + AB + ABb)
    dic = {
        'angle': '%.2f°' % agl,
        'AA': '%.6f%%' % AA,
        'AB': '%.6f%%' % AB,
        'ABb': '%.6f%%' % ABb,
        'GAP': '%.6f%%' % allPercent,
    }
    return [agl, AA, AB, ABb, GAP]


def calLamList(*args):
    Area = []
    res = []
    for val in args[0]:
        angle, length = val[0], val[1]
        truthLength = length - GAP
        S = getFourArea(l=truthLength / (1 + math.sqrt(3)), L=length, agl=angle)
        Area.append(S)
    for val in Area:
        res.append(calPercent(val))
    PSet = []
    AA, AB, ABb = 0.34, 0.345, 0.35
    for val in res:
        print(val)
        p = val[1] * AA + val[2] * AB + val[3] * ABb
        agl = val[0]
        PSet.append([agl, p])
    return np.array(PSet)


def calLam284List(*args):
    Area = []
    res = []
    for val in args[0]:
        angle, length = val[0], val[1]
        truthLength = length - GAP
        S = getFourArea(l=truthLength / (1 + math.sqrt(3)), L=length, agl=angle)
        Area.append(S)
    for val in Area:
        res.append(calPercent(val))
    # for i in range(len(lam284List)-1):
    #     print('%.2f - %.2f = '%(lam284List[i+1][0], lam284List[i][0]),end='')
    #     print(lam284List[i+1][1]-lam284List[i][1])


def takeFirst(param):
    return param[0]


if __name__ == '__main__':
    lamList = [
        [0.88, 92.2339827372934],
        [0.90, 89.774557604498],
        [0.93, 87.3151366893698],
        [1.05, 77.4775032385745],
        [1.21, 67.6399764124984],
        [1.41, 57.8026106511057]
        # [angle, length]
    ]
    lam284List = [
        [1.07, 152.4956114],
        [1.1, 147.5768318],
        [1.14, 142.6580674],
        [1.18, 137.7393197],
        [1.225, 132.8205908],
        [1.27, 127.9018828],
        [1.32, 122.9831981],
        [1.38, 118.0645396]
    ]
    GAP = 9.838
    dataList = []
    newList = lamList + lam284List
    newList.sort(key=takeFirst)
    pSet = calLamList(newList)
    print(pSet)
    # plt.scatter(pSet[:, 0], pSet[:, 1], marker='+')
    # plt.show()
    # calLam284List(lam284List)
    print('finish')
