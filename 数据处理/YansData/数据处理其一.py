#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数据处理其一.py    
@Time    :   2022/4/20 15:10  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   scipy.stats.mode() 用来查找ndarray的众数
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
            if '     697' in line:
                xyz = [line, f.readline()]
            for row in range(2, 699):
                xyz.append([atom.strip() for atom in f.readline().split(' ' * 7)])
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


def drawFig(**kwargs):
    h_ = kwargs['h_']
    o_ = kwargs['o_']
    # 此处为distance计算后的坐标
    fig = plt.figure(dpi=400)
    # 在figure()中增加一个subplot，并且返回axes
    ax = fig.add_subplot(111, projection='3d')
    xh = h_[:, 0]
    yh = h_[:, 1]
    zh = h_[:, 2]

    xo = o_[:, 0]
    yo = o_[:, 1]
    zo = o_[:, 2]
    # print(len(xh), len(xo))  # 273 82
    ax.scatter(xo, yo, zo, linewidth=3, c='red', marker='.')
    ax.scatter(xh, yh, zh, linewidth=1, c='black', marker='.')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


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


def write_xyz_to_file(**kwargs):
    h_ = kwargs['h_']
    o_ = kwargs['o_']

    with open(r'HO.xyz', 'w') as f:
        f.write('     151\n')
        f.write(' i =        0, time =        0.000, E =      -950.0027565385\n')
        for item in o_:
            f.write('  O        %.10f        %.10f        %.10f\n' % (item[0], item[1], item[2]))
        for item in h_:
            f.write('  H        %.10f        %.10f        %.10f\n' % (item[0], item[1], item[2]))


if __name__ == '__main__':
    t1 = time.time()
    FilePath = r'C:\Users\Celeste\Desktop\i0-pos-1.xyz'
    file_a = r'C:\Users\Celeste\Desktop\1Ha.xyz'
    file_b = r'C:\Users\Celeste\Desktop\1Hb.xyz'
    file_c = r'C:\Users\Celeste\Desktop\1Hc.xyz'
    file_f = r'C:\Users\Celeste\Desktop\f-1.xyz'

    # FilePath = r'C:\Users\Celeste\Desktop\testpos.xyz'
    dataSet = read_xyz(file_f)
    # 文件分为697+2行
    count = 0
    with open('1_f.xyz', 'w') as pos:

        for i in dataSet:
            count += 1
            title = i[:2]
            # 写title
            pos.write(i[0])
            pos.write(i[1])

            frameNow = np.array(i[2:699])             # 现在我们获取到了数据

            H_index = np.where(frameNow[:, 0] == 'H')
            O_index = np.where(frameNow[:, 0] == 'O')
            C_index = np.where(frameNow[:, 0] == 'C')
            N_index = np.where(frameNow[:, 0] == 'N')

            # len(H_index[0]) = 327 [172, ..., 498],  len(O_index[0]) = 82 [0, ..., 81]
            # 获取氢原子和氧原子的初始下标位置，优化算法减少计算量, 前 82个为O, 后327个为 H
            # HO_frame = np.concatenate([frameNow[O_index[0]], frameNow[H_index[0]]])[:, 1:].astype(np.float)
            # 获取HO_frame帧的坐标信息并将字符串转为float类型，减少和优化计算量。

            oArray = frameNow[O_index[0]][:, 1:].astype(np.float)
            hArray = frameNow[H_index[0]][:, 1:].astype(np.float)
            cArray = frameNow[C_index[0]][:, 1:].astype(np.float)
            nArray = frameNow[N_index[0]][:, 1:].astype(np.float)
            # 此处为验证O N H C的总数量
            # print('%d >> P_O_N_H_C == %d' % (count, len(oArray)+len(hArray)+len(cArray)+len(nArray)))

            index_row, index_col = calDistance(h_=hArray, o_=oArray)

            # oArr = oArray[index_row[1]]
            # hArr = hArray[index_row[0]]
            # 当前 hArr oArr存储的是几乎邻近的HnO结构坐标信息，变量名复用会产生未知的影响，我们不建议重复使用变量名。
            # print(len(index_row[0]), len(index_row[1]))
            # different frame has different index_row value of index while param == 1.5 value is 151
            # # drawFig(h_=hArray[index_row], o_=oArray[index_col])

            h2o, o2h, max_distance, mod = find_H3O_Structure(h_=hArray, o_=oArray, index_=index_col)
            pos.write('  P        %.10f        %.10f        %.10f\n' % (h2o[0], h2o[1], h2o[2]))
            # write sequence O N H C
            for o in oArray:
                pos.write('  O        %.10f        %.10f        %.10f\n' % (o[0], o[1], o[2]))

            for n in nArray:
                pos.write('  N        %.10f        %.10f        %.10f\n' % (n[0], n[1], n[2]))

            for h in hArray:
                if (h == h2o).all():
                    # if mod.count[0] >= 3:
                    print(count, ' >> ', h, max_distance, mod.count)
                    continue
                pos.write('  H        %.10f        %.10f        %.10f\n' % (h[0], h[1], h[2]))

            for c in cArray:
                pos.write('  C        %.10f        %.10f        %.10f\n' % (c[0], c[1], c[2]))

            # print(count, ' >> ', h2o, o2h, max_distance)  # 正确返回H3O结构中的 最远的H+的坐标

    t2 = time.time()
    print('finish, using time %d s' % (t2 - t1))