# K-平均值算法
# K表示分为多少类：K=3
# 确定类中心：随机生成三个类中心
# 计算相似度：（欧式距离）进行计算
# 更新类中心：new_x = (x1+x2+…)/n1 new_y = (y1+y2+…)/n1,重新划分三个类
#
#############################################################################
# 本代码中的 s, v, c 无特殊声明分别代表Svip, Vip, Common三种用户缩写           #
#############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

def randX():
    x1 = random.randint(10,60)
    x2 = random.randint(10,60)
    x3 = random.randint(10,60)
    return x1,x2,x3
def randY():
    y1 = random.randint(100,500)
    y2 = random.randint(100,500)
    y3 = random.randint(100,500)
    return y1,y2,y3

# def draw(x,y,points):
#     plt.rcParams['font.sans-serif'] = 'SimHei'
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.figure(figsize=(6, 5))
#     plt.scatter(x, y, color='chocolate', linewidths=0.1)
#     plt.scatter(points[:,0],points[:,1], color='limegreen', marker='*')
#     plt.title('消费信息散点图')
#     plt.xlabel('平均消费周期（天）')
#     plt.ylabel('平均每次消费金额')
#     # plt.savefig('消费信息图')
#     # x >> [0,100] , y >> [0,800]
#     plt.show()
#     print('画完了')
def Child():
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False
    return plt.figure(figsize=(16, 9), dpi=100)

def drawpic(class1,class2,class3,points,flag):
    plt.scatter(class1[:, 0], class1[:, 1], color='lightpink')
    plt.scatter(class2[:, 0], class2[:, 1], color='cyan')
    plt.scatter(class3[:, 0], class3[:, 1], color='burlywood')
    plt.scatter(points[:,0],points[:,1], color='limegreen', marker='x')
    plt.title('第%d次迭代'%flag)
    plt.ylabel('平均每次消费金额')
    plt.savefig('消费信息图')
    # x >> [0,100] , y >> [0,800]
    # plt.show()
    print('添加子图%d'%flag)
def K_means(data,):
    pass
def distance(data,points):
    return np.sum((data[0] - points[0]) ** 2+(data[1]-points[1]) ** 2)

def newX(x1,x2,x3):
    return float('%.1f'%(np.sum(x1[:, 0]) / len(x1))),float('%.1f'%(np.sum(x2[:, 0]) / len(x2))),float('%.1f'%(np.sum(x3[:, 0]) / len(x3)))
def newY(y1,y2,y3):
    return float('%.1f'%(np.sum(y1[:, 1]) / len(y1))),float('%.1f'%(np.sum(y2[:, 1]) / len(y2))),float('%.1f'%(np.sum(y3[:, 1]) / len(y3)))

# def new_X(x1,x2,x3):
#     return int(np.sum(x1[:, 0]) / len(x1)),int(np.sum(x2[:, 0]) / len(x2)),int(np.sum(x3[:, 0]) / len(x3))
# def new_Y(y1,y2,y3):
#     return int(np.sum(y1[:, 1]) / len(y1)),int(np.sum(y2[:, 1]) / len(y2)),int(np.sum(y3[:, 1]) / len(y3))


file = pd.read_csv(r'C:\Users\Administrator\Desktop\company.csv',encoding='gbk')
# 周期为x,消费为y
dataSet = file[['平均消费周期（天）','平均每次消费金额']].values
# 现在dataSet里存的是列表中的列表，一对一对的值
# print(dataSet[:10].values) !!!!! 就是你，.values,困扰了我这么久！！！
x1, x2, x3 = randX()
y1, y2, y3 = randY()
points = np.array([[x1,y1],[x2,y2],[x3,y3]]) # 三个随机点
print('初始随机点坐标为：', points[0], points[1], points[2])
# print(points[:,0])
# draw(dataSet[:,0],dataSet[:,1],points) # 每次传参画图
flag = 1
child = Child()
while True:
    tmpclass1 = []
    tmpclass2 = []
    tmpclass3 = []
    for d in dataSet:
        dis1 = distance(d,points[0])
        dis2 = distance(d,points[1])
        dis3 = distance(d,points[2])
        dis = min(dis1,dis2,dis3)
        # 这里print了，返回正确
        if dis == dis1: # 第一类点
            tmpclass1.append([d[0], d[1]])
        elif dis == dis2:  # 第一类点
            tmpclass2.append([d[0], d[1]])
        elif dis == dis3:  # 第一类点
            tmpclass3.append([d[0], d[1]])
    # 为啥列表不能用冒号访问呢？必须用array
    npclass1 = np.array(tmpclass1)
    npclass2 = np.array(tmpclass2)
    npclass3 = np.array(tmpclass3)
    # print("类1：")
    # count = 0
    # for i in npclass1:
    #     count += 1
    #     print(i,end=' ')
    # print('共%d个'%count)
    #
    # count = 0
    # print("类2：")
    # for i in npclass2:
    #     count += 1
    #     print(i,end=' ')
    # print('共%d个' % count)
    # count = 0
    # print("类3：")
    # for i in npclass3:
    #     count += 1
    #     print(i,end=' ')
    # print('共%d个' % count)
    #
    # # 调用新坐标公式
    x1,x2,x3 = newX(npclass1, npclass2, npclass3)
    y1,y2,y3 = newY(npclass1, npclass2, npclass3)
    new_points = np.array([[x1, y1], [x2, y2], [x3, y3]])
    if (points == new_points).all():
        break
    else:
        points = new_points
        child.add_subplot(2, 3, flag)
        drawpic(npclass1, npclass2, npclass3, new_points,flag)
        flag+=1
    # print('最后坐标：', points[0], points[1], points[2])
plt.show()





            # draw(x,y)
# 随机点：svip,vip,common 三种性质，选择是s,v,c代表






