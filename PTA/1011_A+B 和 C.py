#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1011_A+B 和 C.py    
@Time    :   2022/1/4 12:42  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

if __name__ == '__main__':
    num = int(input())
    numList = []
    for i in range(num):
        test = list(map(int, input().split()))
        multi = test[0] + test[1]
        if multi > test[2]:
            test.append('true')
        else:
            test.append('false')
        numList.append(test)
    for i in range(num):
        print('Case #{}: {}'.format(i+1, numList[i][-1]))