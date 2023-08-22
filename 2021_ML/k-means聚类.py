#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   k-means聚类.py    
@Time    :   2021/2/16 21:18  
@Tips    :   聚类算法，欧式距离
             随机生成50个点，然后用k值来聚类
             numpy真是个好东西
'''

import math
import random
import numpy as np
import matplotlib.pyplot as plt

def randPoints(): # 生成90个随机点
    pointList = []
    for l in range(30):
        x = random.randint(0,12)
        y = random.randint(10,22)
        pointList.append([x,y])
    for l in range(30):
        x = random.randint(8,20)
        y = random.randint(0,12)
        pointList.append([x,y])
    for l in range(30):
        x = random.randint(18, 28)
        y = random.randint(15, 28)
        pointList.append([x, y])
    return pointList

def draw():
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 字体
    plt.rcParams['axes.unicode_minus'] = False  # 字符编码
    return plt.figure(figsize=(16, 16), dpi=300)  # 返回一个plt画布

def initial_k(k): # 生成k个初始聚类点
    kPoints = []
    for i in range(k):
        x = random.randint(2, 25)
        y = random.randint(2, 25)
        kPoints.append([x,y])
    return kPoints

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def k_means(pList,initial_k):
    pic = draw()
    kList = []
    picCount = 1
    pic.add_subplot(3, 3, picCount)
    plt.scatter(pList[:,0],pList[:,1])
    plt.scatter(initial_k[:,0],initial_k[:,1],marker='*',color='black')
    koints = initial_k
    while True:
        picCount += 1
        pic.add_subplot(3, 3, picCount)
        dic = {}
        for num in range(len(initial_k)):
            dic[num] = []
        for i in range(len(pList)):
            distance = np.sum((pList[i,:]-koints)**2,axis=1)
            minDistance = np.argmin(distance)
            dic[minDistance].append(pList[i,:])
        tmpDic = {}
        tmpPoints = []
        for k,v in dic.items():
            tmpDic[k] = np.array(v)
            newx = int(np.sum(np.array(v)[:,0])/len(v))
            newy = int(np.sum(np.array(v)[:,1])/len(v))
            tmpPoints.append([newx,newy])
        # 转移类点K
        newKoints = np.array(tmpPoints)
        for k in tmpDic.keys():
            plt.scatter(np.array(tmpDic[k])[:,0],np.array(tmpDic[k])[:,1],color=randomcolor())
        plt.scatter(newKoints[:, 0], newKoints[:, 1], marker='*', color='k')
        plt.title('第%d次迭代'%(picCount-1))
        if (newKoints == koints).all():
            print('第%d次迭代'%(picCount-1),'\n','共%d个聚类点，他们为\n'%(len(newKoints)),newKoints)
            break
        else:
            koints = newKoints
    plt.show()
if __name__ == '__main__':
    pList = np.array(randPoints()) # 90个随机点集
    initial_kList = np.array(initial_k(k=3)) # 三个K聚类点
    k_means(pList=pList,initial_k=initial_kList)
# print(pList[:,0])
# plt.scatter(pList[:,0],pList[:,1],color='red')
# plt.show()
