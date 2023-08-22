#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class8.py    
@Time    :   2021/3/6 16:17  
@Tips    :   基于原子的位置分析O-H键的键长，H-O-H键角的变化趋势
'''
from Cailiaoren.class7 import *
import math
import sys
# if sys.version[0] == 2:
#     input = raw_input
# 注意处理周期性的问题，成键不合理

def unwrap_PBC(current_coord,atom1,atom2):
    xx,yy,zz = current_coord[atom1] - current_coord[atom2]
    if xx > 0.5: xx -= 1
    if xx < -0.5: xx += 1
    if yy > 0.5: yy -= 1
    if yy < -0.5: yy += 1
    if zz > 0.5: zz -= 1
    if zz < -0.5: zz += 1
    return [xx, yy, zz]

def get_current_average_bond(scaling_factor,lattice,atom_set1,atom_set2,current_coord):
    current_average_bond_length = 0.0
    for at1 in atom_set1:
        for at2 in atom_set2:
            if at1 == at2:
                continue
            direct_atom1_atom2 = unwrap_PBC(current_coord,at1,at2)
            lattice = lattice*scaling_factor
            cartesian_atom1_atom2 = np.dot(direct_atom1_atom2,lattice)
            # bond_length = np.linalg.norm(cartesian_atom1_atom2,ord=2)
            bond_length =math.sqrt(sum([math.pow(i,2) for i in cartesian_atom1_atom2]))
            if bond_length < cutoff_O_H:
                continue
            current_average_bond_length  = (current_average_bond_length*index + bond_length)/(index+1)
    return current_average_bond_length

def get_all_possible_angles(scaling_factor,lattice,center_set,side_set1, side_set2,current_coord):
    index = 0
    lattice = lattice*scaling_factor
    current_ave_angle = 0.0
    for i in center_set:
        for j in side_set1:
            if i==j:
                continue
            else:
                direct_at1_at2 = unwrap_PBC(current_coord,i,j)
                cartesian_at1_at2 = np.dot(direct_at1_at2,lattice)
                bond_length1 = np.linalg.norm(cartesian_at1_at2,ard=2)
                if bond_length1 >= cutoff_O_H:
                    continue
            for k in side_set2:
                if i==k or j==k:
                    continue
                else:
                    direct_at1_at3 = unwrap_PBC(current_coord,i,k)
                    cartesian_at1_at3 = np.dot(direct_at1_at3,lattice)
                    bond_length2 = np.linalg.norm(cartesian_at1_at3,ard=2)
                    if bond_length2 >= cutoff_O_H:
                        continue
                angle = math.acos(np.dot(cartesian_at1_at2,cartesian_at1_at3)/(bond_length1*bond_length2))/math.pi*180
                current_ave_angle = (current_ave_angle*index+angle)/(index+1)
                index += 1
    return current_ave_angle

XDATCAR7 = open('data/XDATCAR7','r')
scaling_factor,lattice,all_element = lattice_read(XDATCAR7)
init_coord = coord_read(XDATCAR7,all_element)
# elem1 = input('请输入元素类型，例如：输入‘H’表示为 H（氢元素）：')
# elem2 = input('请输入元素类型，例如：输入‘O’表示为 O（氧元素）：')
center = input()
side1 = input()
side2 = input()
index = 0
center_set = []
side_set1, side_set2 = [], []
atom_set1, atom_set2 = [], []
cutoff_O_H = 1.2 # 氢氧键键长
for i in all_element: # 中心原子和边界原子的下标
    if i == center:
        center_set.append(index)
    elif i == side1:
        side_set1.append(index)
    elif i == side2:
        side_set2.append(index)
    index += 1
# for i in all_element:
#     if i == 'H':
#         atom_set1.append(index)
#     elif i == 'O':
#         atom_set2.append(index)
#     index += 1
init_angle = get_all_possible_angles()
init_bond_length = get_current_average_bond(scaling_factor,lattice,atom_set1,atom_set2,init_coord)
tmp = XDATCAR7.readline()
up_current_average_length_list = [(1.0*0.5,init_bond_length)]
up_current_average_length = init_bond_length
index = 1
NPT = True
if "Direct configuration" in tmp :
    NPT = False
XDATCAR7.seek(XDATCAR7.tell()-len(tmp),0)
while XDATCAR7.tell() < os.path.getsize('data/OSZICAR'):
    if NPT == True:
        scaling_factor,lattice,all_element = lattice_read(XDATCAR7)
    current_coord = coord_read(XDATCAR7,all_element)
    current_bond_length = get_current_average_bond(scaling_factor,lattice,atom_set1,atom_set2,current_coord)
    up_current_average_length = (up_current_average_length*index+current_bond_length)/(index+1)
    up_current_average_length_list.append((index*0.5+0.5,init_bond_length))
    index += 1
with open('data/classBond.txt','w') as w:
    for i in up_current_average_length_list:
        w.write('%f %f\n'%(i[0],i[1]))

print('写完了')