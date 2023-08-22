#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1012_数字分类.py    
@Time    :   2022/1/4 13:07  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def divBy5(n):
    return n % 5


def isEven(n):
    if n % 2 == 0:
        return True
    return False


if __name__ == '__main__':
    numLis = list(map(int, input().split()))
    A1 = [0, False]
    A2 = [0, False]
    A3 = [0, False]
    A4 = [False]
    A5 = [-1, False]
    count = 0
    n = 0
    for i in numLis[1:]:
        if divBy5(i) == 0:
            if isEven(i):
                A1[1] = True
                A1[0] += i
        if divBy5(i) == 1:
            A2[1] = True
            A2[0] += (-1)**n * i
            n += 1
        if divBy5(i) == 2:
            A3[1] = True
            count += 1
            A3[0] = count
        if divBy5(i) == 3:
            A4[0] = True
            A4.append(i)
        if divBy5(i) == 4:
            A5[1] = True
            if A5[0] < i:
                A5[0] = i
    if not A1[1]:
        print('N', end=' ')
    else:
        print(A1[0], end=' ')
    if not A2[1]:
        print('N', end=' ')
    else:
        print(A2[0], end=' ')
    if not A3[1]:
        print('N', end=' ')
    else:
        print(A3[0], end=' ')
    if not A4[0]:
        print('N', end=' ')
    else:
        meanA4 = round(sum(A4[1:]) / (len(A4)-1), 1)
        print(meanA4, end=' ')
    if not A5[1]:
        print('N')
    else:
        print(A5[0])
