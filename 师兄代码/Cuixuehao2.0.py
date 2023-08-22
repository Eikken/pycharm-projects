#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   Cuixuehao2.0.py    
@Time    :   2020/12/18 19:05  
@See     :   现在修改按坐标比率计算
            a:b:c = 3.180:5.508:14.577
'''
import itertools
dicData = {
    'a1':[0.5,0.5,0.5],'a2': [0,0,0.5], 'aa1':[0.5,0.5,0], 'aa2':[0,0,0],
    'b1':[0.5,0.1667,0.4387],'b2':[ 0,0.3333,0.5613],'b3':[ 0,0.6667,0.4387],'b4':[0.5,0.8333,0.5613 ],
    'bb1':[ 0.5,0.1667,0.0291],'bb2':[0,0.3333,0.9709],'bb3':[0,0.6667,0.0291],'bb4':[ 0.5,0.8333,0.9709]
}

def getDicData():
    string = 'ABCDEFGHIJKL'
    keysList = list(dicData.keys())
    for k in keysList:
        dicData[string[0]] = dicData.pop(k)
        string = string.replace(string[0],'',1)
    return dicData

def switchCase(key = ""):
    dicData = getDicData()
    return dicData[key]

def getC2():
    dataSet = []
    dicKeys = []
    for i in itertools.combinations('ABCDEFGHIJKL', 2): # C12_2
        dataSet.append(list(i)) #转成二维list
        s = i[0] + i[1]
        dicKeys.append(s)
    return dataSet,dicKeys

def replaceList(data2List=""):
    temList = []
    # print(l)
    for i in data2List:
        temList.append([switchCase(i[0]), switchCase(i[1])])#, switchCase(i[2])])#转换完成
    return temList

def calDistance(pointList = "", dicKeys = ""):
    #计算所有组坐标点的距离
    dic = {}
    for (i,k) in zip(pointList,dicKeys):
        i[1][0]
        OuShi_dis = (((i[1][0] - i[0][0])*3.180)**2 + ((i[1][1] - i[0][1])*5.508)**2 + ((i[1][2] - i[0][2])*14.577)**2)**0.5
        dic[k] = round(OuShi_dis,6)
    return dic

if __name__ == '__main__':
    data2List,dicKeys = getC2()
    C12_2 = open('C12_2_2.0.txt','w',encoding='utf-8')
    pointList = replaceList(data2List) #把全组合AB换成坐标
    for (p,a) in zip(pointList,data2List):
        C12_2.write(str(a)+'==>'+str(p)+'\n')
    dic = calDistance(pointList,dicKeys)
    v = list(dic.values())
    set_lst = set(v)
    C12_2.write('set 集合长度为' + str(len(set_lst))+'\n')
    for s in set_lst:
        C12_2.write('距离为'+str(s)+'的点有：')
        for k in dic.keys():
            if s == dic[k]:
                C12_2.write(str(k)+' ')
        C12_2.write('\n')
    C12_2.close()

