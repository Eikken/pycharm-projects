#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class1.py    
@Time    :   2021/1/19 19:34  
@Tips    :   Python基础的list、str、数据类型
'''
from copy import copy,deepcopy
myname = 'Celeste'
myage = 17.5
# 体验格式化输出
print('我叫 {0:s},我今年{1:f}岁了。'.format(myname,myage))
print("我的姓名 %s ,我的年龄 %f 岁"%(myname, myage))

a = [[2],3,4,5]
b = a
cpa = copy(a) # 浅层复制，给仓库配了个钥匙cpa
dpa = deepcopy(a) # 深层复制，重新new一个仓库[[2],3,4,5]，钥匙给dpa。
print(a,b,cpa,dpa) #[[2], 3, 4, 5] [[2], 3, 4, 5] [[2], 3, 4, 5] [[2], 3, 4, 5]
a[1] = [123]
print(a,b,cpa,dpa) #[[2], [123], 4, 5] [[2], [123], 4, 5] [[2], 3, 4, 5] [[2], 3, 4, 5]
print(int(round(1.001000))) # 返回最接近的整数
print(type(eval('3.1415926'))) #<class 'float'>
s = 'a string'
ls = list(s)
print(ls) #['a', ' ', 's', 't', 'r', 'i', 'n', 'g']
a = [1,2,3,4]
print(str(a))
print(tuple(a))
print(2 in a)
# 获取list中指定元素下标 ,list 和 str 同样的方式
index = s.index('t')
print(index)

def approString(s):
    try:
        res = float(s)
    except:
        index = s.index('(')
        print(index)
        res = s[0:index]
    return res

ss0 = '192.168(1.2）'
ss1 = '123.1'

print(approString(ss0))
print(approString(ss1))
