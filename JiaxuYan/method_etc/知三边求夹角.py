#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   知三边求夹角.py    
@Time    :   2021/9/6 13:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import math


def qiujiaodu(a_, b_, c_):
    return math.degrees(math.acos((a_ * a_ - b_ * b_ - c_ * c_) / (-2 * b_ * c_)))


def qiuduibian(b_, c_, Agl):
    pass


a, b, c = map(float, '37.52 142 142'.split())
# a, b, c = map(float, '169.28 142 142'.split())  # map(float, input('三条边：').split())
# print(a, '>>', b, '>>', c)
# a=1
# b=1
# c=math.sqrt(2)
AA = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
# a, b, c = a, b, c = map(float, '112.86 142 142'.split())
a, b, c = map(float, '2.236 2 1'.split())
A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
print(A, '||', B, '||', C)
print(A + B + C)

# 46.8307769200258 || 73.17594545975987
# 120.00672237978567

# 73.18599610433455  169.3
# 46.8307769200258 112.86
# print(B)
# print(C)
