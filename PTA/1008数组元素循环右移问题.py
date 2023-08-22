#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1008数组元素循环右移问题.py    
@Time    :   2021/12/13 22:01  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

if __name__ == '__main__':
    n, m = [int(i) for i in input().split()]
    nL = [i for i in input().split()]
    pL = nL
    if m < n:
        if m == 0:
            pL = []
            print(' '.join(nL))
        else:
            for i in nL[-m:]:
                print(i, end=' ')
                pL.remove(i)
    if m > n:
        m = m % n
        if m == 0:
            pL = []
            print(' '.join(nL))
        else:
            for i in nL[-m:]:
                print(i, end=' ')
                pL.remove(i)
    if m == n:
        pL = []
        print(' '.join(nL))
    if pL:
        if len(pL) == 1:
            print(' '.join(pL))
        else:
            print(' '.join(pL))
    # print(pL[-1], end='')
