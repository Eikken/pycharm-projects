#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1001_3n+1猜想.py
@Time    :   2021/11/10 18:25  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def isodd(x):
    if x % 2 != 0:
        return True
    return False


def Callatz(n, count):
    if isodd(n):
        if n == 1:
            return count
        return Callatz((3 * n + 1) / 2, count + 1)
    else:
        return Callatz(n / 2, count + 1)


if __name__ == '__main__':
    x = int(input())
    c = 0
    print(Callatz(x, c))
