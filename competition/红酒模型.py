# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Imputer
import pandas as pd
import numpy as np
import os
from competition.GraModel import GraModel

# baseDir = ''#当前目录
# staticDir = os.path.join(baseDir,'Static')#静态文件目录
# resultDir = os.path.join(baseDir,'Result')#结果文件目录
def fillNaN(dataSet):
    data = np.array(dataSet)
    imp_mean = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imp_mean.fit_transform(data)
    return pd.DataFrame(data)


data = pd.read_excel('灰度表1.xlsx')
columns = ['线路总成本','总里程', '业务类型',
           '需求类型2', '是否续签', '车辆吨位',
           '打包类型','运输等级', '需求紧急程度',
           '标的展示策略','调价紧急程度','成交对象']
data = data.loc[:, columns]
# dataSet = data[columns]
data = fillNaN(data)
data.columns = columns
# print(data)
model = GraModel(data, standard=True)
result = model.result
meanCors = result['meanCors']['value']
# print(result)
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
#可视化矩阵
plt.clf()
plt.figure(figsize=(9,3))
sns.heatmap(meanCors.reshape(1,-1), square=True, annot=True,  cbar=False,
            vmax=1.0,
            linewidths=0.1,cmap='viridis')
plt.yticks([0,],['线路总成本'])
plt.xticks(np.arange(0.5,12,1),columns,rotation=90)
plt.title('关联度可视化矩阵')
plt.savefig('/指标关联度可视化矩阵.png',dpi=100,bbox_inches='tight')
plt.show()