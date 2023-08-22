#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   huaxian.py    
@Time    :   2021/12/5 22:41  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# csv 类似 txt 文件
dataSet = np.array(pd.read_csv(r'csvfile\Sheet2.csv'))#, encoding='gbk') # encoding 声明编码
plt.figure(figsize=(5, 8))
for i in range(1, 26):  # +(i*5000)
    canshu = (i-1) * 100
    plt.plot(dataSet[:, 0], dataSet[:, i] + canshu, linewidth=0.5)
    index = np.where(dataSet[:, i] == dataSet[:, i].max())
plt.savefig('png/nmm14_2.png', dpi=400)
plt.show()
# print(dataSet.head())