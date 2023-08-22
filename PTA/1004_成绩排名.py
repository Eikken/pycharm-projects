#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1004_成绩排名.py    
@Time    :   2021/11/18 20:48  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def takeThird(args):
    return args[2]


def sortStu(*args):
    nList = args[0]
    nList.sort(key=takeThird)
    return nList


if __name__ == '__main__':
    nL = [None for i in range(int(input()))]
    for i in range(len(nL)):
        info = input().split(' ')
        nL[i] = [info[0], info[1], int(info[2])]
    # nL = [['Joe', 'Math990112', 89],
    #       ['Mike', 'CS991301', 100],
    #       ['Mary', 'EE990830', 95]]
    nL = sortStu(nL)
    print(nL[-1][0], end=' ')
    print(nL[-1][1])
    print(nL[0][0], end=' ')
    print(nL[0][1])
