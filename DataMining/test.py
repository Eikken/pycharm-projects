#!/user/bin python
#coding=UTF-8
'''
@author  : Eikken
#@file   : test.py
#@time   : 2019-05-29 15:14:09
'''
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [2, 5, 8, 11, 0]
# intersection
intersection = [v for v in a if v in b]
# union
union = b.extend([v for v in a])
# difference
difference = [v for v in a if v not in b]
print(intersection)
print(difference)