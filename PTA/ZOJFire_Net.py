#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   ZOJFire_Net.py
@Time    :   2022/9/9 17:06  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np

if __name__ == '__main__':
    while True:
        n = int(input())
        if n == 0:
            break
        city = np.zeros((n, n))
        for i in range(n):
            line = list(input())
            for j in range(n):
                if line[j] == 'X':
                    city[i][j] = 1
        city_ = np.array(city)
        for j in np.array(city):
            for k in j:
                pass
                
