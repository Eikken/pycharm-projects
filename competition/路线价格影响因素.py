import numpy as np
import pandas as pd
from competition.ID3 import majorityCnt
from math import log


def getColumns():
    modelData = pd.read_excel(r"historyData.xlsx")
    data = np.array(modelData)
    columns = modelData.columns
    return modelData, data, columns  # 返回 读取数据、pandas数据、DataFrame.columns


def calDicData(dataSet):
    data = dataSet.values.tolist()
    dic = {}
    for i in data:
        dic[i] = dic.get(i, 0) + 1
    return dic

def clearColumns(columns,data):
    # 这一步是把传进来的列名和数据重新组合，返回的数据是一个大集合、去边缘化的数据
    # 包含线路、线路价格等
    col = columns.tolist()
    col.remove("调价比例")
    col.remove("任务id")
    col.remove("是否续签")
    col.remove("调价类型")
    col.remove("交易成功时长")
    col.remove("线路价格（不含税）")
    col.remove("线路指导价（不含税）")
    col.remove("子包号")
    col.remove("线路编码")
    col.remove("运输等级")
    #"时间" in c or "日期" in c or "议价" in c or
    for c in columns:
        if "次数" in c or "网点" in c:
            col.remove(c)
    tmpCol = col
    for c in col:
        dic = calDicData(data[c])
        if len(dic) == 1:
            tmpCol.remove(c)
    #返回一信息度最高的columns 的 list
    tmpCol.append("线路价格（不含税）")
    tmpCol.append("线路编码")
    dataSet = data.loc[:, tmpCol]
    return tmpCol, dataSet

def deleteSameData(dic):
    for c in dic.columns.values.tolist():
        if dic == 1:
            continue
        else:
            print(dic[[c]])

def getEffectiveData( DataFrame, dic):
    # 我们获取每一条路线的所有去边缘化信息，进行决策树规划，发现价格的影响因素
    # modelData 是所有路线的总数据，本函数返回某条路线的所有数据。
    print('筛选路线>>')
    for key in dic.keys():
        if dic[key] == 1:  # key 是字典中存的所有路线，对于只出现一次的路线，舍去
            continue
        else:
            dic[key] = DataFrame[DataFrame['线路编码'] == key]
    return dic # 字典现在存的是{路线:所有该路线信息}

def dropSame(col, dataSet):
    for c in col:
        flag = True
        var = dataSet[[c]][0]
        for d in dataSet[[c]]:
            if d == var:
                continue
            else:
                flag = False
                break
        if not flag:
            col.remove(c)
            dataSet.drop(columns=[c])
    return dataSet
# def creatDicTree(routeDic):
#     for k, v in routeDic:
#         print("线路", k, "的决策树为：")
#         print(v.columns)
if __name__ == '__main__':

    modelData, data, columns = getColumns()
    col, dataSet = clearColumns(columns, modelData) # dataSet二维数组
    dic = calDicData(modelData["线路编码"])
    routeDic = getEffectiveData(dataSet, dic)
    for k in routeDic:
        deleteSameData(routeDic[k])
        pass
    # col, dataSet = dropSame(col, np.array(dataSet))
    # print(type(routeDic[85]))<class 'pandas.core.frame.DataFrame'>
    # dic = calDicData(modelData["线路编码"])

    # creatDicTree(routeDic)