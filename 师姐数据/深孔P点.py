#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   深孔P点.py    
@Time    :   2021/5/9 21:57  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
from matplotlib import pyplot as plt


a, b, c, d = 0.0054, 0.0116, -0.3569, -0.0028
x = np.arange(0, 20, 1)
x2 = np.arange(0, 14, 1)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='polar')
# for i in range(20):
#     # for j in range(3):
y1 = np.sin(x*np.pi/180)
y2 = np.sin((x-1)*np.pi/180)
ax1.plot(x, y1-y2)
ax1.plot(x, 2*(y1-y2))
y1 = np.sin(x2*np.pi/180)
y2 = np.sin((x2-1)*np.pi/180)
ax1.plot(x2, 3*(y1-y2))
plt.show()
print('finish')