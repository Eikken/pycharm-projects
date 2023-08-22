#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class6.py    
@Time    :   2021/3/4 16:55  
@Tips    :   lammps数据处理
1、得到石墨烯原胞的基矢或晶格信息
2、得到原胞内所有原子的分数坐标
3、平移操作得到扩胞后周期性的原子
'''

import math
import numpy as np

a = (2.522, 0, 0)
b = (2.522 / 2, 2.522 / 2 * math.sqrt(3), 0)
c = (0, 0, 20)
super_x = 10
super_y = 10  # 扩大十倍
super_z = 1
transformation = np.array([[super_x, 0, 0],
                           [0, super_y, 0],
                           [0, 0, super_z]])
lattice = np.array([a, b, c])
newLattice = np.dot(lattice, transformation)  # 扩胞

C1 = [0, 0, 0.5]
C2 = [1 / float(3), 1 / float(3), 0.5]
# 扩胞后就有100个C1和100个C2
file1 = open('data/map.in','w')
file1.write('%d %d %d 2'%(super_x,super_y,super_z)) # 基矢
file1.write('\n#\\By Celeste Young\n')

allAtoms = []
index = 1
for i in range(super_x):
    for j in range(super_y):
        newC1 = [(0 + i) / float(super_x), (0 + j) / float(super_y), 0.5]
        allAtoms.append(newC1)
        file1.write('%d %d %d 0 %d\n'%(i, j, 0, index))
        index += 1
        newC2 = [(1 / float(3) + i) / float(super_x), (1 / float(3) + j) / float(super_y), 0.5]
        allAtoms.append(newC2)
        file1.write('%d %d %d 1 %d\n'%(i, j, 1, index))
        index += 1
newAtoms = np.dot(np.array(allAtoms),newLattice)
file1.close()
print(len(newAtoms))
# print(newAtoms)
# with open('data/class6.data','w') as writer:
#     writer.write('Celeste Young\n')
#     writer.write('\n')
#     writer.write('%d atoms\n'%(len(allAtoms)))
#     writer.write('2 atom types\n')
#     writer.write('\n')
#     writer.write('%7.3f %7.3f xlo xhi\n'%(0.0,newLattice[0][0])) # 第一行第一列
#     writer.write('%7.3f %7.3f ylo yhi\n'%(0.0,newLattice[1][1]))
#     writer.write('%7.3f %7.3f zlo zhi\n'%(0.0,newLattice[2][2]))
#     writer.write('%7.3f %7.3f %7.3f xy xz yz\n'%(newLattice[1][0],0.0,0.0))
#     writer.write('Masses\n')
#     writer.write('\n')
#     writer.write('1 32.06\n')
#     writer.write('2 54.9380\n')
#     writer.write('Atoms\n')
#     writer.write('\n')
#     index = 0
#     for i in newAtoms[2/3*len(newAtoms)]:
#         writer.write('%d 1 %7.3f %7.3f %7.3f\n'%(index,i[0],i[1],i[2]))
#         index += 1