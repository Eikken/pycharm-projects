
import numpy as np
import pandas as pd
from competition.ID3 import majorityCnt
from math import log

def getColumns():
    modelData = pd.read_excel(r"灰度表.xlsx")
    data = np.array(modelData)
    columns = modelData.columns
    return modelData, data, columns
def calDicData(dataSet):
    # data = np.array(dataSet.loc[:, columnList])
    data =dataSet.values.tolist()
    # data = dataSet['线路指导价（不含税）'].values.tolist()
    # count =0.0
    # lenList = len(data)
    # for d in data:
    #     if d != "普通":
    #         count += 1.0
    # dic = dict.fromkeys(data, 0)
    dic = {}
    for i in data:
        dic[i] = dic.get(i, 0) + 1
    return dic

def getEffectiveData(md):
    print('数据清洗>>')
    # labels = md.columns[2:].tolist() # 2到42列是测试信息数据，我们取其columns
    y = md.columns[5:6].tolist()# 判断标签在第五列价格
    # columnList = labels + y # 合并将判断标签放在最后一个
    dataSet = md.loc[:10, y]# loc是切片获取数据
    return dataSet
def clearColumns(columns,data):
    col = columns.tolist()
    col.remove("调价比例")
    col.remove("任务id")
    col.remove("是否续签")
    col.remove("调价类型")
    col.remove("交易成功时长")
    col.remove("线路价格（不含税）")
    col.remove("线路指导价（不含税）")
    col.remove("线路编码")
    for c in columns:
        if "时间" in c or "日期" in c or "议价" in c:
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


##分割数据集
def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集
    :param axis:划分数据集的特征的维度
    :param value:特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """

    # 循环遍历dataSet中的每一行数据
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] # 删除这一维特征
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

##计算信息熵
# 计算的始终是类别标签的不确定度
def calcShannonEnt(dataSet):
    """
    计算训练数据集中的Y随机变量的香农熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet) # 实例的个数
    labelCounts = {}
    for featVec in dataSet: # 遍历每个实例，统计标签的频次
        currentLabel = featVec[-1] # 表示最后一列
        # 当前标签不在labelCounts map中，就让labelCounts加入该标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel] +=1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob,2) # log base 2
    return shannonEnt

## 计算条件熵
def calcConditionalEntropy(dataSet,i,featList,uniqueVals):
    """
    计算x_i给定的条件下，Y的条件熵
    :param dataSet: 数据集
    :param i: 维度i
    :param featList: 数据集特征列表
    :param unqiueVals: 数据集特征集合
    :return: 条件熵
    """
    ce = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet,i,value)
        prob = len(subDataSet) / float(len(dataSet)) # 极大似然估计概率
        ce += prob * calcShannonEnt(subDataSet) #∑pH(Y|X=xi) 条件熵的计算
    return ce

##计算信息增益
def calcInformationGain(dataSet,baseEntropy,i):
    """
    计算信息增益
    :param dataSet: 数据集
    :param baseEntropy: 数据集中Y的信息熵
    :param i: 特征维度i
    :return: 特征i对数据集的信息增益g(dataSet | X_i)
    """
    featList = [example[i] for example in dataSet] # 第i维特征列表
    uniqueVals = set(featList) # 换成集合 - 集合中的每个元素不重复
    newEntropy = calcConditionalEntropy(dataSet,i,featList,uniqueVals)#计算条件熵，
    infoGain = baseEntropy - newEntropy # 信息增益 = 信息熵 - 条件熵
    return infoGain

## 算法框架
def chooseBestFeatureToSplitByID3(dataSet):
    """
    选择最好的数据集划分
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1 # 最后一列是分类
    baseEntropy = calcShannonEnt(dataSet) #返回整个数据集的信息熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures): # 遍历所有维度特征
        infoGain = calcInformationGain(dataSet,baseEntropy,i) #返回具体特征的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature # 返回最佳特征对应的维度

def createTree(dataSet,featureName,chooseBestFeatureToSplitFunc = chooseBestFeatureToSplitByID3):
    """
    创建决策树
    :param dataSet: 数据集
    :param featureName: 数据集每一维的名称
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet] # 类别列表
    if classList.count(classList[0]) == len(classList): # 统计属于列别classList[0]的个数
        return classList[0] # 当类别完全相同则停止继续划分
    if len(dataSet[0]) ==1: # 当只有一个特征的时候，遍历所有实例返回出现次数最多的类别
        return majorityCnt(classList) # 返回类别标签
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)#最佳特征对应的索引
    bestFeatLabel = featureName[bestFeat] #最佳特征
    myTree ={bestFeatLabel:{}}  # map 结构，且key为featureLabel
    del (featureName[bestFeat])
    # 找到需要分类的特征子集
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = featureName[:] # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


if __name__ == '__main__':
    modelData, data, columns = getColumns()
    # col, dataSet = clearColumns(columns, modelData)
    # # 测试决策树的构建
    # # dataSet, featureName = createDataSet()  # data == dataSet and col==featureName
    # print(col)
    # dic = calDicData(modelData["线路编码"])
    # print(sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    # col = modelData.columns.values.tolist()
    columns = ['总里程', '业务类型', '需求类型1',
               '需求类型2', '是否续签', '车辆长度', '车辆吨位', '打包类型',
               '运输等级', '需求紧急程度',
               '计划运输时长', '线路总成本','线路价格（不含税）']
    dataSet = modelData.loc[:, columns]
    myTree = createTree(np.array(dataSet).tolist() , columns)
    print(myTree)
    # print(len(col)) 46
    # print(len(columns))63
    # dataSet = getEffectiveData(data)
    # print(dataSet)
    # dic = calPuTong(data)
    # print(len(dic))
    # for k, v in dic.items():
    #     print('keys:{:<2}  dig:{}'.format(k, v))

    # print("不是普通的值有：",count,"个")
    # print("占比率为：",round(count/lenList,20),"%")
    # print("lenlist:",lenList)
    print("-------------------")
    # print(type(data))
    # print(columns)