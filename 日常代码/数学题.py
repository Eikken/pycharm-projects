#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数学题.py    
@Time    :   2021/8/20 15:18  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   数学题
'''

import random

res = []

for i in range(100):
    a = random.randint(10, 90)
    b = random.randint(10, 800)
    c = a * b
    print('(%d) : %d * %d = ?' % (i, a, b))
    res.append(['(%d) : %d * %d = %d' % (i, a, b, c)])

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

for i in res:
    print(i[0])