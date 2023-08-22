#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   积分测试.py    
@Time    :   2021/5/10 20:21  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
from scipy import integrate


def f(x):
    return 1 / x


k = integrate.quad(f, 1, 2)

print(k)
