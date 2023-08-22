#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1003我要通过.py    
@Time    :   2021/11/10 21:08  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   https://pintia.cn/problem-sets/994805260223102976/problems/994805323154440192
'''


def isA(s):
    for i in range(len(s)):
        if s[i] != "A":
            return False
    return True


def isCorrect(ls):
    if 'P' not in ls or 'A' not in ls or 'T' not in ls:
        return 'NO'
    if 'P' in ls and 'T' in ls:
        if ls.count('P') == 1 and ls.count('T') == 1:
            if ls.index('P') < ls.index('T'):
                if ls.split('P')[0] == '':
                    if ls.split('P')[1].split('T')[1] == '' and len(ls.split('P')[1].split('T')[0]) >= 1:
                        if isA(ls.split('P')[1].split('T')[0]):
                            return "YES"
                        return "NO"
                    return "NO"
                else:
                    if isA(ls.split('P')[0]) and ls.split('P')[1].split('T')[0] != '':
                        if isA(ls.split('P')[1].split('T')[1]) and len(ls.split('P')[1].split('T')[1]) \
                                == len(ls.split('P')[0]) * len(ls.split('P')[1].split('T')[0]):
                            return "YES"
                        return "NO"
                    return "NO"
                    # if len(ss[0]) * len(ss[1]) == len(ss[2]):
                    #     return 'YES'
            return 'NO'
        return "NO"
    return "NO"


if __name__ == '__main__':

    x = int(input())
    a = [None for i in range(x)]
    for i in range(x):
        a[i] = input()
    for i in a:
        print(isCorrect(i))
