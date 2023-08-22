#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   移动火柴棍.py    
@Time    :   2022/10/22 17:30  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def moveGun(a_, b_, c_):
    count_ = 0
    bigger = a_ if a_ > b_ else b_
    smaller = a_ if a_ < b_ else b_

    print(c_ - bigger, c - smaller)
    return count_


if __name__ == '__main__':

    # import sys
    #
    # lines = sys.stdin.readlines()

    # allLines = [item.split('\n') for item in lines]

    allLines = [['4', ''],
                ['123+321=444', ''],
                ['088+111=999', ''],
                ['808+111=991', ''],
                ['009+004=009', ''],
                ['', '']]
    sampleLength = int(allLines[0][0])
    sampleData = [i[0] for i in allLines[1:sampleLength+1]]
    for i in sampleData:
        a = int(i.split('+')[0])
        b = int(i.split('+')[1].split('=')[0])
        c = int(i.split('+')[1].split('=')[1])
        if a + b == c:
            print(0)
        else:
            count = moveGun(a, b, c)
