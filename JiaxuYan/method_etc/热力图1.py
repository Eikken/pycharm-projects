#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   热力图1.py    
@Time    :   2021/12/16 20:00  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.random.rand(8, 6)
dataSet = pd.read_excel(r'E:\桌面文件备份\twist\angle_LBM5-15.xls', sheet_name='Sheet3')
heatMap = np.array(dataSet)[17:, 17:]
columns = dataSet.columns
print(columns)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(9, 9))
# 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
# 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
# sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['a', 'b', 'c'], index = range(1,5)),
#                 annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="YlGnBu")
sns.heatmap(heatMap, annot=False, vmax=0.2, vmin=0, xticklabels=columns[17:], yticklabels=columns[17:],
            square=True, cmap="YlGnBu")
ax.set_title('热力图', fontsize=18)
ax.set_ylabel('angle', fontsize=18)
ax.set_xlabel('angle', fontsize=18)  # 横变成y轴，跟矩阵原始的布局情况是一样的
plt.savefig('png/pingaddfig/热力图17-.png', dpi=300)
plt.show()
