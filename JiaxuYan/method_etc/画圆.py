#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   画圆.py    
@Time    :   2021/3/15 22:36  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import math


def f(x):
    return (1 - (x - 1) ** 2) ** 0.5


# v, err = integrate.quad(f, 0, 0.5)
# print(v*4)

#
x1 = np.linspace(0, 2, 200)
x2 = np.linspace(-1, 1, 200)
y1 = np.sqrt(1 - (x1 - 1) ** 2)

fig = plt.figure(figsize=(6, 4), dpi=100)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x1, f(x1), color='blue')
# ax1.plot(x1,,color='blue')
# ax1.plot(x2,y1,color='red')
# ax1.plot(x2,-y1,color='red')
# ax.fill_between(x, f(x), color='green', alpha=0.5)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='grey')
plt.axvline(x=0.5, ymin=0, ymax=1, linestyle='--', color='grey')
# plt.axvline(x=0,ymin=0,ymax=1,linestyle='--',color='grey')
# plt.axvline(x=1,ymin=0,ymax=1,linestyle='--',color='grey')
# c1 = plt.Circle((0,0),1)
# plt.gcf().gca().add_artist(c1)
plt.show()
