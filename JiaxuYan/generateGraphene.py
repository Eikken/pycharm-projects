#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   generateGraphene.py    
@Time    :   2021/3/17 15:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   石墨烯原胞扩胞
键长2.522A

'''
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
import pandas as pd


def matrixTransformation(x_, y_, theta):
    Matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    xT, yT = [], []
    for k, v in zip(x_, y_):
        twistMatrix = np.dot([k, v], Matrix)
        # 矩阵 1X2 * 2X2 = 1X2
        xT.append(twistMatrix[0])
        yT.append(twistMatrix[1])
    return np.array(xT), np.array(yT)


a = (2.460, 0, 0)
b = (2.460 / 2, 2.460 / 2 * math.sqrt(3), 0)
c = (0, 0, 20)
# 扩胞矩阵
super_x = 5
super_y = 5
super_z = 1

transformtion = np.array([[super_x, 0, 0],
                          [0, super_y, 0],
                          [0, 0, super_z]])

lattice = np.array([a, b, c])
newLattice = np.dot(lattice, transformtion)
# print(newLattice)  # 晶胞扩大

C1 = [0, 0, 0.5]
C2 = [1 / float(3), 1 / float(3), 0.5]
Frac1 = 0
Frac2 = 1 / float(3)
allAtoms = []
for i in range(super_x):
    for j in range(super_y):
        newC1 = [(Frac1 + i) / super_x, (Frac1 + j) / super_y, 0.5]
        newC2 = [(Frac2 + i) / super_x, (Frac2 + j) / super_y, 0.5]
        allAtoms.append(newC1)
        allAtoms.append(newC2)
newAllAtoms = np.dot(np.array(allAtoms), newLattice)
# print(newAllAtoms)

# with open('data/graphene.data', 'w') as writer:
#     writer.write('Graphene By Celeste\n\n')
#     writer.write('%d atoms\n' % (len(allAtoms)))
#     writer.write('1 atom types\n\n')
#     writer.write('%7.3f %7.3f xlo xhi\n' % (0.0, newLattice[0][0]))
#     writer.write('%7.3f %7.3f ylo yhi\n' % (0.0, newLattice[0][0]))
#     writer.write('%7.3f %7.3f zlo zhi\n' % (0.0, newLattice[0][0]))
#     writer.write('%7.3f %7.3f %7.3f xy xz yz\n' % (newLattice[1][0],0.0,0.0))
#     writer.write('  Masses\n\n')
#     writer.write('1 12.0107\n\n')
#     writer.write('Atoms\n\n')
#     index = 1
#     for i in newAllAtoms:
#         writer.write('%d 1 %7.3f %7.3f %7.3f\n'%(index,i[0],i[1],i[2]))
#         index += 1

xList = np.array(newAllAtoms).T[0]
yList = np.array(newAllAtoms).T[1]
zList = np.array(newAllAtoms).T[2]
x_mean = np.mean(xList)
y_mean = np.mean(yList)
xList = xList - x_mean
yList = yList - y_mean
# print(np.mean(xList),',',np.mean(yList))
# 9331.4 , 5387.486301916073
# 20210330
s1 = np.stack((xList, yList), axis=-1)
dis = distance.cdist(s1, s1)
# print(dis)
# numpy to txt
# np.savetxt('data/5X5_distance.txt', dis)
# numpy to pandas to xlsx
df = pd.DataFrame(dis)
# df.to_excel('data/5X5_distance.xlsx', index=False, header=False)
fig = plt.figure(figsize=(9, 5), edgecolor='black')
plt.subplot(111)
plt.scatter(xList, yList, 5, marker='.', color='green')
plt.scatter([0.0], [0.0], 10, marker='*', color='red')
plt.xticks([])
plt.yticks([])
plt.savefig('png/扩胞展示图_simple_green.png', dpi=100)
plt.show()
print('finish')
# print(doublePointsDistance([mox[0], moy[0]], [mox[1], moy[1]]))
# mox,moy是去除area之外的剩下的points集合，mo = 模
# angle = float(10)
# thetaAngle = np.pi * angle / 180.0
# xTwist, yTwist = matrixTransformation(mox, moy, thetaAngle)
# Index = calOverLapCircle(mox, moy, xTwist, yTwist)
# s1 = np.stack((mox, moy), axis=-1)
# s2 = np.stack((xTwist, yTwist), axis=-1)
# dis = distance.cdist(s1, s2)
# index = np.where(dis.min(axis=0) < 14)
# # plt.scatter(s1[index[0],0], s1[index[0],1], 10, marker='.', color='green')
# # plt.scatter(s2[index[1],0], s2[index[1],1], 10, marker='.', color='red')
# plt.scatter(mox, moy, 10, marker='.', color='green')
# plt.scatter(xTwist, yTwist, 10, marker='.', color='red')
# plt.scatter(0, 0, 10, marker='*', color='black')
# # index = np.where(distance.cdist(s1, s2).min(axis=0) == distance.cdist(s1, s2))
# # print(distance.cdist(s1, s2).min(axis=1) == distance.cdist(s1, s2))
# # print(s1[index[0]],'\n\n',s2[index[0]])
# # # s1 是列column，s2是行column
# print("showed %d points, all %d points" % (len(index[0]),len(s1)))
# print(index)
# print(dis.min(axis=1))
# plt.show()
