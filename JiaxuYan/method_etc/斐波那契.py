#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   斐波那契.py    
@Time    :   2021/1/15 20:21  

@Tips    :

'''
def Feiboonci(fn1,fn2):
    return fn1 + fn2

L = 1
S = 1
fb = ['L','S']
print('1>> 1 >>L')
for i in range(2,10):
    print(i,end='>> ')
    fn = Feiboonci(L,S)
    S = L
    L = fn
    print(fn,end=' >>')
    lis = []
    for l in fb:
        print(l,end='')
        if l == 'L':
            lis.append('L')
            lis.append('S')
        else:
            lis.append('L')
    fb = lis
    print()