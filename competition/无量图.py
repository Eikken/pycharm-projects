import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['font.serif'] = ['KaiTi']

from competition.ID3 import majorityCnt
from math import log
# 灰色关联结果矩阵可视化
# 灰色关联结果矩阵可视化
import seaborn as sns

# 无量纲化
def dimensionlessProcessing(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        MEAN = d.mean()
        newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()
    return newDataFrame

def GRA_ONE(gray, m=0):
    # 读取为df格式
    gray = dimensionlessProcessing(gray)
    # 标准化
    std = gray.iloc[:, m]  # 为标准要素
    gray.drop(str(m),axis=1,inplace=True)
    ce = gray.iloc[:, 0:]  # 为比较要素
    shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c, d = np.amax(a), np.amin(a)

    # 计算值
    result = np.zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值,并返回
    result_list = [np.mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m,1)
    return pd.DataFrame(result_list)


def GRA(DataFrame):
    df = DataFrame.copy()
    list_columns = [
        str(s) for s in range(len(df.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    col = df.columns
    df.columns=list_columns
    for i in range(len(df.columns)):
        df_local.iloc[:, i] = GRA_ONE(df, m=i)[0]
    df_local.columns = col
    return df_local



def ShowGRAHeatMap(DataFrame):
    colormap = plt.cm.RdBu
    ylabels = DataFrame.columns.values.tolist()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    f, ax = plt.subplots(figsize=(14, 14))
    ax.set_title('灰度表')

    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    mask=mask,
                    )
    plt.show()

def getColumns():
    modelData = pd.read_excel(r"灰度表1.xlsx", encoding="utf-8")
    data = np.array(modelData)
    col = modelData.columns.tolist()
    col.remove("计划靠车时间")
    col.remove("计划到达时间")
    return modelData.loc[:, col], data, col # 返回 读取数据、pandas数据、DataFrame.columns

def fillNaN(dataSet):
    data = np.array(dataSet)
    columns = np.array(dataSet.columns[:len(dataSet.columns) - 2])
    imp_mean = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imp_mean.fit_transform(data)
    return pd.DataFrame(data), columns.tolist()

if __name__ == '__main__':
    modelData, data, columns = getColumns()
    # print(columns)
    columns =[ '线路总成本','总里程', '业务类型',
           '需求类型2', '是否续签', '车辆长度', '车辆吨位',
           '需求紧急程度','成交对象',
            '调价紧急程度','标的展示策略']
    # columns = ['线路价格（不含税）', '总里程', '业务类型', '需求类型1',
    #            '需求类型2', '是否续签', '车辆长度', '车辆吨位', '打包类型',
    #            '运输等级', '需求紧急程度',  '计划卸货等待时长',
    #            '计划运输时长', '线路总成本']
    data = modelData.loc[:, columns]
    dataSet, col = fillNaN(data)
    dataSet.columns = columns
    data_wine_gra = GRA(dataSet)
    ShowGRAHeatMap(data_wine_gra)
