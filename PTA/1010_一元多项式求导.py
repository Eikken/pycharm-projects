#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   1010_一元多项式求导.py    
@Time    :   2022/1/4 11:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

if __name__ == '__main__':
    lis = list(map(int, input().split()))
    res = []
    for i in range(len(lis) // 2):
        ii = i * 2
        jj = i * 2 + 1
        if lis[jj] == lis[ii] == 0:
            res.append(str(lis[ii]))
            res.append(str(lis[jj]))
            continue
        elif lis[jj] == 0:
            # res.append(str(lis[jj]))
            # res.append(str(lis[jj]))
            continue
        else:
            res.append(str(lis[ii] * lis[jj]))
            res.append(str(lis[jj] - 1))
    if not res:
        print('0 0')
    else:
        print(' '.join(res))
