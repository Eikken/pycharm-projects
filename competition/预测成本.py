import os

import xlrd
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import Imputer
import xlwt
from xlutils.copy import copy

plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False
plt.style.use("ggplot")
plt.rcParams['figure.figsize'] = (16, 16)
def fillNaN(dataSet):
    data = np.array(dataSet)
    imp_mean = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imp_mean.fit_transform(data)
    return pd.DataFrame(data)

def dataWrite(lis,lie):
    dir = os.path.abspath('.').split('src')[0]
    oldWb = xlrd.open_workbook(dir + "\\成本预测值附件.xls")
    w = copy(oldWb)
    for i in range(len(lis)):
        w.get_sheet(0).write(i + 1, lie, round(lis[i]))
    w.save('成本预测值附件.xlsx')
    print("file saved")

advert = pd.read_excel("灰度表1.xlsx")
dataSet = pd.read_excel("预测表1.xlsx")
columns = ['线路总成本','总里程', '业务类型', '需求类型1',
           '需求类型2', '是否续签', '车辆长度', '车辆吨位',
           '打包类型','运输等级', '需求紧急程度','计划运输时长',
           '标的展示策略','成交对象']
advert = advert[columns]
dataSet = dataSet[columns[1:]]
# dataSet = fillNaN(dataSet)
# dataSet.columns = columns[1:]
advert = fillNaN(advert)
advert.columns = columns
col = columns[1:]
X = advert[col]
y = advert['线路总成本']
lm1 = LinearRegression()
lm1.fit(X, y)
lm1_predict = lm1.predict(X[col])
# print(lm1.intercept_)
nparr = lm1.coef_.tolist() # 模型值转list好计算
lis = []
dataSet.columns = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
for i in range(1489):
    count = 0.0
    for j in range(13):
        count += dataSet[str(j)][i]*nparr[j]
    lis.append(count+lm1.intercept_)
dataWrite(lis, 5)


# print(type(lm1.coef_)) # numpy.ndarray
# for i in lm1.coef_:
#     print(i)
# print("R^2:",r2_score(y,lm1_predict))

