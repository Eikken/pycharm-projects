#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   两个方向变化求长度.py    
@Time    :   2022/12/11 21:57  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def rtTri(*args, **kwargs):
    return (args[0]**2 + args[1]**2)**0.5


def inZigzag(*args):
    # 例如 9.9466 = 9.9465 == 994 = 994
    if int(args[0]*100) in (length_zigzag[:, 1]*100).astype(int):
        return True
    return False


if __name__ == '__main__':
    # start here

    a = 2.46
    acc = a / 3 ** 0.5
    length_zigzag = []
    for i in range(1, 17):
        longH = i*a/2
        shortH1 = acc/2
        shortH2 = acc
        # i%2==0 means shortH2
        if i%2==0:
            shortH = shortH2
            length_zigzag.append([i, rtTri(longH, shortH)])
            # print(i, longH, shortH, length)
        else:
            shortH = shortH1
            length_zigzag.append([i, rtTri(longH, shortH)])
            # print(i, longH, shortH, length)
    length_zigzag = np.array(length_zigzag)
    print(length_zigzag)
    print()
    length_armchair = []
    for i in range(1, 17):

        if i%3==0:
            longH = acc * i
            longH_up = longH + acc/2
            longH_down = longH - acc/2
            shortH = acc*np.cos(np.pi/6)
            l1 = rtTri(longH_down, shortH)
            l2 = rtTri(longH_up, shortH)
            # 没有3的点，但是有边长为3i*acc ± acc/2, 高为acc*np.cos(30)的点，一对一对
            if inZigzag(l1):
                print('is in Zigzag', [i, l1, 1])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l1, 1])
            if inZigzag(l2):
                print('is in Zigzag', [i, l2, 1])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l2, 1])  # 高为0的沿armchair边

            shortH = a + acc*np.cos(np.pi/6)
            l3 = rtTri(longH_down, shortH)
            l4 = rtTri(longH_up, shortH)

            if inZigzag(l3):
                print('is in Zigzag', [i, l3, 3])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l3, 3])
            if inZigzag(l4):
                print('is in Zigzag', [i, l4, 3])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l4, 3])

        else:
            longH = acc*i
            l1 = rtTri(longH, 0)
            # length_armchair.append([i, l1, 0])  # 高为0的沿armchair边
            if inZigzag(l1):
                print('is in Zigzag', [i, l1, 0])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l1, 0])
            shortH = a
            l2 = rtTri(longH, shortH)
            # length_armchair.append([i, l2, 2])  # 高为a的沿armchair边
            if inZigzag(l2):
                print('is in Zigzag', [i, l2, 2])
                pass  # 高为0的沿armchair边
            else:
                length_armchair.append([i, l2, 2])
    print()
    length_armchair = np.array(length_armchair)
    print(length_armchair)

    print(a / (3**0.5*2*np.sin(np.deg2rad(4.41)/2)))