#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   AandB.py    
@Time    :   2022/9/9 16:19  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

if __name__ == '__main__':
    import sys
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        [a, b] = map(int, line.split())
        print(a + b)

