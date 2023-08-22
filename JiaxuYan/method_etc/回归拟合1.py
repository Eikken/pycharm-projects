#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   回归拟合1.py    
@Time    :   2021/11/28 22:58  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataSet142 = np.array([
    [0.84, 32.46450907334302, 10.821503024447674, 37.48678610520727, 9.838],
    [0.93, 31.645485247063505, 10.548495082354501, 36.54105885205689, 9.838],
    [1.05, 30.63326561040845, 10.211088536802816, 35.372248292653246, 9.838],
    [1.21, 29.350952083722593, 9.783650694574197, 33.89156017301809, 9.838],
    [1.41, 27.675192537784042, 9.22506417926135, 31.956559723128684, 9.838]
])

dataSet284 = np.array([
    [1.07, 35.17376978017918, 11.724589926726395, 40.61517090200075, 9.838],
    [1.1, 35.01225935679811, 11.670753118932703, 40.42867472916876, 9.838],
    [1.14, 34.84002287569277, 11.613340958564256, 40.22979317170787, 9.838],
    [1.18, 34.6559569459445, 11.551985648648166, 40.01725214353028, 9.838],
    [1.225, 34.458801774364964, 11.486267258121657, 39.78959696076314, 9.838],
    [1.27, 34.24711247087119, 11.415704156957062, 39.54515920804973, 9.838],
    [1.32, 34.01922385589248, 11.339741285297494, 39.28201610164332, 9.838],
    [1.38, 33.773206939575736, 11.25773564652525, 38.99794023592198, 9.838]
])

x = dataSet142[:, 0].reshape(len(dataSet142[:, 0]), 1)
y = dataSet142[:, 2].reshape(len(dataSet142[:, 2]), 1)

print(x)
# xx = dataSet284[:, 0].reshape(len(dataSet284[:, 0]), 1)
# yy = dataSet284[:, 1].reshape(len(dataSet284[:, 1]), 1)
#
# reg = LinearRegression().fit(xx, yy)
# print("一元回归方程为:  Y = %.5fX + %.5f" % (reg.coef_[0][0], reg.intercept_[0]))
# print("R平方为: %s" % reg.score(xx, yy))
#
# plt.scatter(xx, yy, color='black')
# plt.plot(xx, reg.predict(xx), color='red', linewidth=1)
# # plt.scatter(np.array([0.1]).reshape(1, 1), reg.predict(np.array([0.1]).reshape(1, 1)), color='red', marker='+')
# plt.show()
# print('col=2:predict[0.1] = %f' % reg.predict(np.array([0.1]).reshape(1, 1))[0][0])
