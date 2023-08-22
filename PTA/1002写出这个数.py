#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1002写出这个数.py    
@Time    :   2021/11/10 18:41  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


def calSum(s):
    addList = []
    for i in range(len(s)):
        addList.append(int(s[i]))
    return sum(addList)


def num2py(n):
    dic = {
        '0': 'ling', '1': 'yi', '2': 'er', '3': 'san',
        '4': 'si', '5': 'wu', '6': 'liu', '7': 'qi',
        '8': 'ba', '9': 'jiu'
    }
    return dic[str(n)]


def readNumber(n, nL):
    if n // 10 == 0:
        nL.append(num2py(n))
        return nL
    else:
        num = n % 10
        nL.append(num2py(num))
        n = n//10
        return readNumber(n, nL)


if __name__ == '__main__':
    x = input()
    noneList = []
    readNumber(calSum(x), noneList).reverse()
    if len(noneList) == 1:
        print(noneList[0])
    else:
        for i in range(len(noneList)-1):
            print(noneList[i], end=' ')
        print(noneList[-1], end='')

