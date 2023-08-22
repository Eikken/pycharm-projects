#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   CuiXuehaoData.py    
@Time    :   2020/12/15 22:28  
'''
# a1（0.5,0.5,0.5） a2(0,0,0.5) aa1 (0.5,0.5,0) aa2(0,0,0)
# b1(0.5,0.13,0.452) b2(0,0.335,0.548) b3(1,0.706,0.452) b4(0.5,0.83,0.548)
# bb1(0.5,0.13,0.952) bb2(0,0.335,0.048) bb3(1,0.706,0.952) bb4(0.5,0.83,0.048)
# 使用a单写表示上方原子，aa双写表示下方原子，b同理，一共 12 个point
import itertools
from collections import Counter
dicData = {
    'a1':[0.5,0.5,0.5],'a2': [0,0,0.5], 'aa1':[0.5,0.5,0], 'aa2':[0,0,0],
    'b1':[0.5,0.1667,0.4387],'b2':[ 0,0.3333,0.5613],'b3':[ 0,0.6667,0.4387],'b4':[0.5,0.8333,0.5613 ],
    'bb1':[ 0.5,0.1667,0.0291],'bb2':[0,0.3333,0.9709],'bb3':[0,0.6667,0.0291],'bb4':[ 0.5,0.8333,0.9709]
}
# 1-范数，计算方式为向量所有元素的绝对值之和。
#
# 2-范数，计算方式跟欧式距离的方式一致

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
        #i =('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'E'), ('A', 'F'), ('A', 'G'), ('A', 'H'), ('A', 'I'), ('A', 'J'), ('A', 'K'), ('A', 'L'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('B', 'F'), ('B', 'G'), ('B', 'H'), ('B', 'I'), ('B', 'J'), ('B', 'K'), ('B', 'L'), ('C', 'D'), ('C', 'E'), ('C', 'F'), ('C', 'G'), ('C', 'H'), ('C', 'I'), ('C', 'J'), ('C', 'K'), ('C', 'L'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('D', 'H'), ('D', 'I'), ('D', 'J'), ('D', 'K'), ('D', 'L'), ('E', 'F'), ('E', 'G'), ('E', 'H'), ('E', 'I'), ('E', 'J'), ('E', 'K'), ('E', 'L'), ('F', 'G'), ('F', 'H'), ('F', 'I'), ('F', 'J'), ('F', 'K'), ('F', 'L'), ('G', 'H'), ('G', 'I'), ('G', 'J'), ('G', 'K'), ('G', 'L'), ('H', 'I'), ('H', 'J'), ('H', 'K'), ('H', 'L'), ('I', 'J'), ('I', 'K'), ('I', 'L'), ('J', 'K'), ('J', 'L'), ('K', 'L')
        dataSet.append(list(i)) #转成二维list
        s = i[0] + i[1]
        dicKeys.append(s)
        # print(''.join(i),end=",")
    return dataSet,dicKeys

def replaceList(data2List=""):
    temList = []
    # l = len(data2List[0])
    # print(l)
    for i in data2List:
        temList.append([switchCase(i[0]), switchCase(i[1])])#, switchCase(i[2])])#转换完成
    return temList

def calDistance(pointList = "", dicKeys = ""):
    #计算所有组坐标点的距离
    dic = {}
    for (i,k) in zip(pointList,dicKeys):
        OuShi_dis = ((i[1][0] - i[0][0])**2 + (i[1][1] - i[0][1])**2 + (i[1][2] - i[0][2])**2)**0.5
        dic[k] = round(OuShi_dis,6)
    # for (i,k) in zip(pointList,dicKeys):
    #     a = ((i[1][0] - i[0][0])**2 + (i[1][1] - i[0][1])**2 + (i[1][2] - i[0][2])**2)**0.5
    #     b = ((i[2][0] - i[0][0])**2 + (i[2][1] - i[0][1])**2 + (i[2][2] - i[2][2])**2)**0.5
    #     c = ((i[1][0] - i[2][0])**2 + (i[1][1] - i[2][1])**2 + (i[1][2] - i[2][2])**2)**0.5
    #     # 三条边的距离求三角形面积相等
    #     a = round(a, 6)
    #     b = round(b, 6)
    #     c = round(c, 6)
    #     s = (a + b + c) / 2
    #     S = 0.0
    #     if a!=0.0 and b!=0.0 and c!=0.0:
    #         S = (s * (s - a) * (s - b) * (s - c)) ** 0.5
    #     # print('a ',a,',b ',b,',c ',c,'\n',k,':',S)
    #     dic[k] = S
    # list1 = list(dic.values())
    # dict_cnt = {}  # dict_cnt=dict()
    # for item in list1:
    #     if item in dict_cnt:  # 直接判断key在不在字典中
    #         dict_cnt[item] += 1
    #     else:
    #         dict_cnt[item] = 1
    # dictt = sorted(dict_cnt.items(), key=lambda x: x[1], reverse=True)
    # print(dictt)
    # print(len(dictt))
    # dictt = dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
    return dic

if __name__ == '__main__':
    data2List,dicKeys = getC2()
    C12_2 = open('C12_2.txt','w',encoding='utf-8')
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
        # print(s,'出现了',Counter(dic.values())[s],'次')
    C12_2.close()

    # # set会生成一个元素无序且不重复的可迭代对象，也就是我们常说的去重
    # if len(set_lst) == len(v):
    #     print('列表里的元素互不重复！')
    # else:
    #     print('列表里有重复的元素！')
    # count = 0
    # for k,v in dic.items():
    #     flag = 0
    #     for k2,v2 in dic.items():
    #         if v == v2 and flag == 0:
    #             print('距离为'+str(v)+'的点有',end=':')
    #             print(k2,end=' ')
    #             flag = 1
    #             count += 1
    #         elif v == v2 and flag == 1:
    #             print(k2, end=' ')
    #     flag = 0
    #     print()
    # print(count)
    # switchCase('B')
