#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   反求角度.py    
@Time    :   2021/6/30 15:51  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   已知伪半径,已知旋转底边长度,能不能逆推theta
             a*a=b*b+c*c-2bc*cosA
             A = arccos(b*b+c*c-a*a / 2bc)
'''

import numpy as np
import matplotlib.pyplot as plt


def getTheta(a_, L):  # L底边长；a腰长
    cosA = (2 * a_ ** 2 - L ** 2) / (2 * a_ ** 2)
    # print(cosA)
    return np.rad2deg(np.arccos(cosA))


def CosineTheorem(a_, Theta):
    Theta = np.deg2rad(Theta)
    return (2.0 * a_ ** 2 * (1 - np.cos(Theta))) ** 0.5


if __name__ == '__main__':
    cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
                  '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
                  '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
                  '27.8': 512.0898359, '29.41': 1398.815212957022}
    a = 142.0  # 281662
    a2 = 246
    b = 142.0281662 * np.cos(np.deg2rad(30))
    d = 0.5 * CosineTheorem(a, 120)    # 单个石墨烯内一条边上的高
    # print(getTheta(2*l1, 2*l1)) # print(getTheta(l1, l1))  # 60.00
    # print(getTheta(751, 2*a))  # 21.798340810544445
    # print(getTheta(1024, 4 * d))  # 27.795123516182407
    # print(d)
    # print(60 - getTheta(3601.47, 8 * d))
    # print(60 - getTheta(1238, 8 * d))  # 13.176464941518184
    Lb = 132.82054
    di = 2.84
    print(getTheta(Lb, di))