#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   热力图fig.py    
@Time    :   2022/5/23 9:37  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   牛姐热力图
'''

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normHeat(heatMap):
    up = heatMap < -258.225
    down = heatMap > -257
    return up == down


filePath = r'C:\Users\Celeste\Desktop\energy.xlsx'

file_data = pd.read_excel(filePath, sheet_name='3l')
heatMap = np.array(file_data)[:, 1:].astype(np.float)
columns = list(file_data.columns)[1:]

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9, 9))

sns.heatmap(heatMap + 256.048713, xticklabels=[], yticklabels=[], annot = True,
            square=True, linewidths='0.2', cmap="bwr", mask=normHeat(heatMap))

plt.show()
