#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数据处理其二.py    
@Time    :   2022/5/9 20:20  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import distance


def read_xyz(filename):
    xyzALl = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line == '':
                break
            if '697' in line:
                xyz = [line, f.readline()]
            for row in range(2, 699):
                xyz.append([atom.strip() for atom in f.readline().split(' ' * 3)[:4]])
            xyzALl.append(xyz)
    return xyzALl


def calDistance(**kwargs):
    h_ = kwargs['h_']
    o_ = kwargs['o_']
    dis1 = distance.cdist(h_, o_, 'euclidean')  # .min(axis=1)
    dis2 = distance.cdist(o_, h_, 'euclidean')  # .min(axis=0)
    #  min.(axis = )none：整个矩阵; 0：每列; 1：每行;
    # df = pd.DataFrame(distance.cdist(h_, o_, 'euclidean'))  # 数据转Excel
    # distance 分为4种状态，(0.95 0.95 1.05), (0.95 0.95 1.85), (0.95 1.85 1.85), (0.95 2.15 2.65)
    # df.to_excel('dataPackage/distance.xlsx', index=True, header=True)
    minParam = 1.6
    # 参数设置在1.5的时候，生成比较合理的H2O或HO结构比较合理
    index_1 = np.where(dis1 < minParam)
    index_2 = np.where(dis2 < minParam)

    # 每个O附近至少有一个H
    return index_1, index_2


def find_H3O_Structure(**kwargs):
    idx_ = kwargs['index_']
    h_ = kwargs['h_']  # [idx_[1]]
    o_ = kwargs['o_']  # [idx_[0]]  # O对H的数组，
    # 使用np.where 返回的distance数组的众数来了找O与H组合最多的即一个O对用三个H。
    mod_ = stats.mode(idx_[0])
    # print(mod_)
    idx_mod_index = np.where(idx_[0] == mod_[0][0])
    idx_h_ = idx_[1][idx_mod_index]
    idx_o_ = idx_[0][idx_mod_index]
    h2o_ = h_[idx_h_]  # H to O coordinates
    o2h_ = o_[idx_o_]  # O to H coordinates
    max_dis = distance.cdist(h2o_, o2h_).max()  # right distance of H3O_+
    max_index = np.where(distance.cdist(h2o_, o2h_) == max_dis)
    # print(distance.cdist(h2o_, o2h_))
    # 此处返回H3O结构中的 最远的H+的坐标
    # print(distance.cdist(h2o[max_index[0]], o2h[max_index[1]]))

    return h2o_[max_index[0]][1, :], o2h_[max_index[1]][1, :], max_dis, mod_


if __name__ == '__main__':
    t1 = time.time()
    file_f = r'C:\Users\Celeste\Desktop\f-1.xyz'
    file_f_2 = r'C:\Users\Celeste\Desktop\f-2.xyz'
    # FilePath = r'C:\Users\Celeste\Desktop\testpos.xyz'
    dataSet = read_xyz(file_f_2)

    count = 0
    with open('2_f.xyz', 'w') as pos:
        for i in dataSet:
            count += 1
            title = i[:2]
            # 写title
            pos.write(i[0])
            pos.write(i[1])

            frameNow = np.array(i[2:699])  # 现在我们获取到了数据

            H_index = np.where(frameNow[:, 0] == 'H')
            O_index = np.where(frameNow[:, 0] == 'O')
            C_index = np.where(frameNow[:, 0] == 'C')
            N_index = np.where(frameNow[:, 0] == 'N')

            oArray = frameNow[O_index[0]][:, 1:].astype(np.float)
            hArray = frameNow[H_index[0]][:, 1:].astype(np.float)
            cArray = frameNow[C_index[0]][:, 1:].astype(np.float)
            nArray = frameNow[N_index[0]][:, 1:].astype(np.float)

            index_row, index_col = calDistance(h_=hArray, o_=oArray)
            h2o, o2h, max_distance, mod = find_H3O_Structure(h_=hArray, o_=oArray, index_=index_col)
            pos.write('  P     %.10f     %.10f     %.10f   \n' % (h2o[0], h2o[1], h2o[2]))

            for h in hArray:
                if (h == h2o).all():
                    # if mod.count[0] >= 3:
                    print(count, ' >> ', h, max_distance, mod.count)
                    continue
                pos.write('  H     %.10f     %.10f     %.10f   \n' % (h[0], h[1], h[2]))

            for c in cArray:
                pos.write('  C     %.10f     %.10f     %.10f   \n' % (c[0], c[1], c[2]))

            for n in nArray:
                pos.write('  N     %.10f     %.10f     %.10f   \n' % (n[0], n[1], n[2]))

            for o in oArray:
                pos.write('  O     %.10f     %.10f     %.10f   \n' % (o[0], o[1], o[2]))

    t2 = time.time()
    print('finish, using time %d s' % (t2 - t1))