#!/user/bin/env python
# coding=utf-8
'''
@author  : Eikken
#@file   : ApiTest.py
#@time   : 2019-05-21 11:57:04
'''
import pandas as pd
import numpy as np


def getDataSet():
    DataSet = pd.read_excel(r'historyData.xlsx', encoding='UTF-8')
    dataSet = np.array(DataSet).tolist()
    columns = np.array(DataSet.columns).tolist()
    data = []
    for d in dataSet:
        d[1] = str(d[1]).split(' ')  # str(d[1]).split(' ') 元素集转化为单个
        data.append(d[1])
    return data, columns


def createItems(dataSet):
    Items = []
    for d in dataSet:
        for item in d:
            if not [item] in Items:  # list 用 not[item] in list
                Items.append([item])
    Items.sort()
    return map(frozenset, Items)


def createSupportItem(D, Items, MinSupport):
    X = {}
    dataSet = list(D)
    items = list(Items)
    sumItem = float(len(dataSet))
    # map对象用一次就空了,所以转化为list
    for d in dataSet:
        for item in items:  # 候选集
            if item.issubset(d):  # 候选集为item子集
                if not item in X:
                    X[item] = 1  # 不存在就创建，存在就加一
                else:
                    X[item] += 1
    supportItems = []  # 返回结果
    supportData = {}
    for k in X.keys():
        support = X[k] / float(sumItem)  # 支持度
        if support >= MinSupport:
            supportItems.insert(0, k)
        supportData[k] = support
    return supportItems, supportData


def AprioriConf(Lk, k):  # 计算K频繁项集
    # Lk 是上一个频繁项集 last
    # k是创建的项集数
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList


def Apriori(dataSet, minSupport):
    Items = createItems(dataSet)
    D = map(set, dataSet)
    L1, supportData = createSupportItem(D, Items, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = AprioriConf(L[k - 2], k)
        Lk, Supk = createSupportItem(map(set, dataSet), Ck, MinSupport=minSupport)
        supportData.update(Supk)
        L.append(Lk)
        k += 1
    return L, supportData


def main():
    dataSet, columns = getDataSet()  # dataSet中仅有项目集，没有订单集
    L, Support = Apriori(dataSet, 0.5)
    print('所有频繁项集L:')
    for l in L:
        print(l)
    print('对应支持度Support:')
    for k, v in Support.items():
        print('项目集：', k, '的支持度为：', v)


if __name__ == '__main__':
    main()

#我现在要选择每一条公线路，他们的价格支持度