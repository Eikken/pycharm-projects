#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   不同维度数组合并.py    
@Time    :   2022/11/23 17:37  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
    先return 一个mid points segments array
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter


def get_segment_midpoint(segments_):
    seg_list = []
    for seg in segments_:
        x_ = sum(seg[:, 0]) / 2
        y_ = sum(seg[:, 1]) / 2
        seg_list.append([x_, y_])

    return np.array(seg_list)


if __name__ == '__main__':
    # start here
    x = np.arange(1, 6)
    y = np.arange(1, 6) ** 2

    xy = np.stack([x, y, x, y], axis=1)
    print(xy)
    # points = np.array([x, y]).T.reshape(-1, 1, 2)
    # # segments = np.concatenate((xy, 2*y.reshape(6, 1)), axis=1)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # segList = get_segment_midpoint(segments)
    # color = "blue"
    # width = 2
    # lwidths = 1 / x
    # rgba_list = [colorConverter.to_rgba(color, alpha=np.abs(lwidth / (width + 0.001))) for lwidth in lwidths]
    # print(rgba_list)
    # fig, a = plt.subplots()
    # plt.tight_layout(pad=2.19)
    # plt.axis('tight')
    # plt.gcf().subplots_adjust(left=0.17)
    #
    # lc = LineCollection(
    #     segments,
    #     linewidths=[2] * len(x),
    #     colors=rgba_list)
    # a.add_collection(lc)
    # plt.show()
    # seg = np.concatenate((segList, segList), axis=0)
    # print(seg)
