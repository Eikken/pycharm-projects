#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   极坐标数据拟合.py    
@Time    :   2021/3/9 15:53  
@Tips    :   
'''

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv(r'txtdata/Sheet1.txt')
clo = np.array(['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7'])
x = (data['l1']/180)*np.pi
y = data['Normalized5'] # ['Normalized1']
# max(y) = 815
y1 = (1*(np.cos(x)**2) + 1*(np.sin(x)**2) + 1*(np.sin(2*x)))**2
for i,j,k in zip(x,y,y1):
    print(i,j,k)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1, projection='polar') # 指定绘制极坐标图
ax1.plot(x,y,color='blue')
# ax1.plot(x,y1,color='red')
plt.show()
print('finish')
