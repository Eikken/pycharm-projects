#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   mn_k1k2.py    
@Time    :   2022/7/2 16:23  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import numpy as np
from itertools import combinations
import math


def arcsin_and_arccos(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    sin = delta_y / math.sqrt(delta_x ** 2 + delta_y ** 2)
    cos = delta_x / math.sqrt(delta_x ** 2 + delta_y ** 2)
    if sin >= 0 and cos >= 0:
        return math.asin(sin), math.acos(cos)
    elif sin >= 0 and cos < 0:
        return math.pi - math.asin(sin), math.acos(cos)
    elif sin < 0 and cos < 0:
        return math.pi - math.asin(sin), 2 * math.pi - math.acos(cos)
    elif sin < 0 and cos >= 0:
        return 2 * math.pi + math.asin(sin), 2 * math.pi - math.acos(cos)


def cos_AB(ab1_, ab2_):
    # (x1 * x2 + y1 * y2) / [√(x1 ^ 2 + y1 ^ 2) *√(x2 ^ 2 + y2 ^ 2)]
    x1, y1 = [i for i in ab1_]
    x2, y2 = [i for i in ab2_]
    cosVal = (x1 * x2 + y1 * y2) / np.linalg.norm(ab1_) * np.linalg.norm(ab2_)
    # print(x1, y1, ' || ', x2, y2)
    # print(np.linalg.norm(ab1_), np.linalg.norm(ab2_))
    if cosVal > 1 or cosVal < -1:
        return 0

    return np.rad2deg(np.arccos(cosVal))


def cal_m_n(m_, n_):
    result = [i for i in range(2, 16)]

    left = m_ ** 2 + n_ ** 2 - m_ * n_
    # print(left)
    if left in result:
        if str(left) not in dictRes.keys():
            dictRes[str(left)] = []
            dictRes[str(left)].append([m_, n_])
        else:
            dictRes[str(left)].append([m_, n_])
        # print("m=%d  n=%d  & k1*k2=%d" % (m_, n_, left))


def cal_mn_12(dR):
    for k, v in dR.items():
        # print('right', k, ' : ', v)
        combList = list(combinations(v, 2))
        for i in combList:
            m_n_1 = np.array(i[0])  # [m1, n1]
            m_n_2 = np.array(i[1])  # [m2, n2]
            ab_1 = [m_n_1[0] * a[0] + m_n_1[1] * b[0], m_n_1[0] * a[1] + m_n_1[1] * b[1]]  # ab_1 由m1a+n1b 构成
            ab_2 = [m_n_2[0] * a[0] + m_n_2[1] * b[0], m_n_2[0] * a[1] + m_n_2[1] * b[1]]  # ab_2 由m2a+n2b 构成
            # theta = np.rad2deg(arcsin_and_arccos(ab_1, ab_2))
            vector_dot_product = np.dot(ab_1, ab_2)
            arccosVal = np.arccos(vector_dot_product / (np.linalg.norm(ab_1) * np.linalg.norm(ab_2)))
            theta = np.degrees(arccosVal)
            if abs(theta-60) < 1 or abs(theta-120) < 1:
                print(k, i, theta)
                # pass


if __name__ == '__main__':
    dictRes = {}
    a = [1, 0]
    b = [-1 / 2, np.sqrt(3) / 2]  # a b crystal vector\

    for m in range(-10, 11):
        for n in range(-10, 11):
            cal_m_n(m, n)

    cal_mn_12(dictRes)
