#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   函数周期测试.py    
@Time    :   2021/3/9 14:17  
@Tips    :   
'''

import numpy as np
from matplotlib import pyplot as plt
import math
a, b, c, d = 0.0054, 0.0116, -0.3569, -0.0028
# a, b, d = 1, 2, 3
# x = np.linspace(-50,50,100)
# y = np.sin(x)(a*(math.cos(x)**2) + b*(math.sin(x)**2) + d*(math.sin(2*x)))**2
#
# plt.plot(x,y,color='red')
x = np.arange(0, 360, 1)
x = x*np.pi/180
y1 = (a*(np.cos(x)**2) + b*(np.sin(x)**2) + d*(np.sin(2*x)))**2
# plt.plot(pi, y)
# plt.show()

y2 = (((b-a)/2)*np.sin(2*x)+d*np.cos(2*x))**2
theta = [1, 2, 3, 4, 5, 6, 7]
y3 = (b*(np.cos(x)**2) + a*np.sin(x)**2 + d*np.sin(2*x))**2
y4 = (((a-b)/2)*np.sin(2*x)+d*np.cos(2*x))**2
y5 = (c*np.cos(x)**2 + a*np.sin(x)**2)**2
y6 = ((c-a)*np.cos(x)*np.sin(x))**2
y7 = (np.sin(2*x))**4
fig = plt.figure()

# ax1 = fig.add_subplot(2, 2, 1, projection='polar') # 指定绘制极坐标图
# ax1.plot(x, y1)
# ax1 = fig.add_subplot(2, 2, 2, projection='polar') # 指定绘制极坐标图
# ax1.scatter(x, y1,1)
ax1 = fig.add_subplot(1, 1, 1)#, projection='polar')
ax1.plot(x, y7)
# ax1 = fig.add_subplot(2, 2, 1, projection='polar') # 指定绘制极坐标图
# ax1.plot(x, y1)
# ax1 = fig.add_subplot(2, 2, 2, projection='polar')
# ax1.plot(x, y2)
# ax1 = fig.add_subplot(2, 2, 3, projection='polar')
# ax1.plot(x, y5)
# ax1 = fig.add_subplot(2, 2, 4, projection='polar')
# ax1.plot(x, y6)
# ax1 = fig.add_subplot(1, 2, 2, projection='polar') # 指定绘制极坐标图
# ax1.scatter(x, y2,1)
# ax2 = fig.add_subplot(1, 2, 2)

# step4: 绘图



# ax2.plot(x, y2, '--')

# 展示
plt.savefig('png/fig5.png',dpi=600)
plt.show()