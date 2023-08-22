#!/user/bin python
#coding=utf-8
'''
@author  : Eikken
#@file   : 大作业.py
#@time   : 2019-05-22 19:27:27
'''
import pandas as pd
import numpy as np
from math import log
import time
import operator
from sklearn.preprocessing import Imputer
import xlwt

def getData():
    modelData = pd.read_csv(r'model.csv')
    testData = pd.read_csv(r'test.csv')
    return modelData,testData

def fillNaN(dataSet):
    data = np.array(dataSet)
    columns = np.array(dataSet.columns[:len(dataSet.columns) - 1])
    imp_mean = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = imp_mean.fit_transform(data)
    return data.tolist(),columns.tolist()

def getEffectiveData(md):
    print('数据清洗>>')
    labels = md.columns[2:].tolist() # 2到42列是测试信息数据，我们取其columns
    y = md.columns[0:1].tolist()  # 判断标签在第一列
    columnList = labels + y # 合并将判断标签放在最后一个
    dataSet = md.loc[:, columnList]# loc是切片获取数据
    print('---------------------------------------------------------------')
    return dataSet

def getTestData(td):
    print('测试数据清洗>>')
    labels = td.columns[1:].tolist()  # 1到41列是测试信息数据，我们取其columns
    dataSet = td.loc[:, labels]  # loc是切片获取数据
    print('---------------------------------------------------------------')
    return np.array(dataSet).tolist()

def dataToList(dataSet):
    return  np.array(dataSet).tolist(), \
            np.array(dataSet.columns[:len(dataSet.columns)-1]).tolist() # 最后一列是判断label

def getCalResult(testResult):
    dic = {}
    for k in testResult:
        dic[k] = dic.get(k, 0) + 1
    return dic
# 计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for feaVec in dataSet:
        currentLabel = feaVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 因为数据集的最后一项是标签
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别相同则停止划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 所有特征已经用完
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 为了不改变原始列表的内容复制了一下
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = float('0.0')
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def classifyAll(inputTree, featLabels, testDataSet):
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll

def printResult(resDict,res):
    a = resDict[0]
    b = resDict[1]
    a_percent = float(a) / len(res)
    print('一共',a+b,'个测试值\n','预测正确：',a,'预测错误：',b)
    print('测试值的正确率：', str(a_percent * 100) + '%')

def getCompare(trueResult, testResult):
    boolList = []
    for (tr, te) in zip(trueResult, testResult):
        if tr==te:
            boolList.append(0) # 相同是0，不相同是1
        else:
            boolList.append(1)
    return boolList

def dataWrite(lis):
    f = xlwt.Workbook()
    sheet = f.add_sheet(u'sheet',cell_overwrite_ok=True)
    sheet.write(0, 0, '201601020729')
    for i in range(len(lis)):
        sheet.write(i + 1, 0, lis[i])
    f.save('datamining.xlsx')

def main():
    t1 = time.clock()
    md,td = getData() # 分为modeldata 和 testdata, getData仅返回我们的整个测试集。
    dataSet = getEffectiveData(md) # 数据处理为有效数据,共10017个数据，且全部为有效数据。清洗数据集，对于Nan值的清除
    fillData, labels = fillNaN(dataSet[:8000]) # fillData 就是填充了NaN值的数据，直接有效。fillData是numpy
    dataList = fillData # 到这一步没有问题。
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels
    desicionTree = createTree(dataList, labels_tmp)
    print('决策树:\n', desicionTree)
    print('---------------------------------------------------------------')
    testDataList,tmplabel = fillNaN(dataSet[8000:10000])  # 取2k个作为测试
    tmplabels = tmplabel[:]
    trueResult = np.array(dataSet.loc[8000:10000,'y']).tolist() # 取2k个他们的结果
    testResult = classifyAll(desicionTree, tmplabels, testDataList)
    print('model测试结果集:\n', testResult)
    print('---------------------------------------------------------------')
    testData = getTestData(td)
    testRes = classifyAll(desicionTree, labels, testData)
    print('test测试结果集:\n', testRes)
    print('---------------------------------------------------------------')
    res = getCompare(trueResult,testResult)
    resDict = getCalResult(res) # 字典存储结果集中的0和1的个数
    printResult(resDict,res)
    print(resDict)
    dataWrite(testRes)
    t2 = time.clock()
    print('测试共耗时:',t2-t1)

if __name__ == '__main__':
    main()


