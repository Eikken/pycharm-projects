#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   list操作合集.py    
@Time    :   2021/9/16 21:28  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

# def myfont():
#     name = input("输入你的名字:(only English words) \t")
#     length = len(name)
#
#     for x in range(0, length):
#         c = name[x]
#         c = c.upper()
#
#         if c == 'A':
#             print('''-----A-----
# ---A---A---
# --A-A-A-A--
# -A-------A-''', '\n')
#
#         elif c == 'B':
#             print('''---B-B-B---
# ---B--B----
# ---B--B----
# ---B-B-B---''', '\n')
#
#         elif c == 'C':
#             print('''---C-C-C---
# --C--------
# --C--------
# ---C-C-C---''', '\n')
#
#         elif c == 'D':
#             print('''---D-D-D---
# ---D----D--
# ---D----D--
# ---D-D-D---''', '\n')
#
#         elif c == 'E':
#             print('''---E-E-E---
# ---EEE-----
# ---EEE-----
# ---E-E-E---''', '\n')
#
#         elif c == 'F':
#             print('''---F-F-F---
# ---F-------
# ---F-F-F---
# ---F-------''', '\n')
#
#         elif c == 'G':
#             print('''---G--GG---
# --G--------
# --G---GG---
# ---G--GG---''', '\n')
#
#         elif c == 'H':
#             print('''--H-----H--
# --H--H--H--
# --H--H--H--
# --H-----H--''', '\n')
#
#         elif c == 'I':
#             print('''--II-I-II--
# -----I-----
# -----I-----
# --II-I-II--''', '\n')
#
#         elif c == 'J':
#             print('''-----J-----
# -----J-----
# --J--J-----
# ---J-J-----''', '\n')
#
#         elif c == 'K':
#             print('''---K---K---
# ---K-K-----
# ---K-K-----
# ---K---K---''', '\n')
#
#         elif c == 'L':
#             print('''--L--------
# --L--------
# --L--------
# --L-L-L-L--''', '\n')
#
#         elif c == 'M':
#             print('''--M-----M--
# --M-M-M-M--
# --M--M--M--
# --M-----M--''', '\n')
#
#         elif c == 'N':
#             print('''--N-----N--
# --N-N---N--
# --N--N--N--
# --N---N-N--''', '\n')
#
#         elif c == 'O':
#             print('''----OOO----
# --OO---OO--
# --OO---OO--
# ----OOO----''', '\n')
#
#         elif c == 'P':
#             print('''---P-P-P---
# ---P----P---
# ---P-P-P----
# ---P--------''', '\n')
#
#         elif c == 'Q':
#             print('''----QQQ----
# --QQ---QQ--
# --QQ-Q-QQ--
# ----QQQ--Q-''', '\n')
#
#         elif c == 'R':
#             print('''--R-RR-----
# --R---R----
# --R-RR-----
# --R---R----''', '\n')
#
#         elif c == 'S':
#             print('''----SS-----
# --SS---SS--
# -SS---SS---
# ----SS-----''', '\n')
#
#         elif c == 'T':
#             print('''--TT-T-TT--
# -----T-----
# -----T-----
# -----T-----''', '\n')
#
#         elif c == 'U':
#             print('''--U-----U--
# --U-----U--
# --U-----U--
# ---U-U-U---''', '\n')
#
#         elif c == 'V':
#             print('''--V-----V--
# ---V---V---
# ----V-V----
# -----V-----''', '\n')
#
#         elif c == 'W':
#             print('''-W---W---W-
# --W--W--W--
# ---W---W---
# ----W-W----''', '\n')
#
#         elif c == 'X':
#             print('''--X-----X--
# ----X-X----
# ----X-X----
# --X-----X--''', '\n')
#
#         elif c == 'Y':
#             print('''--Y-----Y--
# ---Y---Y---
# -----Y-----
# -----Y-----''', '\n')
#
#         elif c == 'Z':
#             print('''--Z--Z--Z--
# -------Z---
# ----Z------
# --Z--Z--Z--''', '\n')
#
#         elif c == ' ':
#             print('''-----------
# -----------
# -----------
# -----------''', '\n')
#
#         elif c == '.':
#             print('''----..-----
# ---..-..---
# ---..-..---
# ----..-----''', '\n')
# 
# if __name__ == '__main__':
#     myfont()
# # li = ["a", "b", "mpilgrim", "z", "example"]
# # print(li)
# # print('{0} >> '.format(li[4]), li[1])  # 下标索引访问
# # print('%s %s ' % ('负数索引', '====数索引'), li[-2])  # 负数索引
# # print('{0} >> '.format('切片访问'), li[1:3])  # 切片访问
# # print('{0} >> '.format('切片访问'), li[2:-1])
# # li.append("new")  # 增加元素
# # print('{0} >> '.format('增加元素'), li)
# # li.append(["newList", 'append list']) # 增加任意元素
# # print('{0} >> '.format('增加任意元素'), li)
# # print('{0} >> '.format('搜索元素'), li.index("example"))  # 搜索元素
# # li.remove("a")
# # print('{0} >> '.format('删除元素'), li)  # 删除元素
# # li = li + ['example', 'new']  # 运算符操作
# # print('{0} >> '.format('运算符操作'), li)
# #
# # numList = ['1', '2', '3', '4', '5']
# # print('{0} >> '.format('jion操作'), '+'.join(numList))  # jion操作
# # print('CHCCHCCCH'.split('H'))
# # numList = [elem*2 for elem in numList]
# # print('{0} >> '.format('list映射操作'), numList)  # list映射操作
#
# a = [1, '1.2', 8, '"9.01"', [1, 3.33]]
# # print(a)
# # print(type(x), type(y), type(z))
# # print(x*2, y*2, int((z.split('.')[0]))*2)
# di = {'name': 'zhangsan', 'age': 18, 'lis': [1, 2, 'qweqwe'], 'zidian': {'a': 1, 'b': 2}}
# print(di['zidian']['b'])
# # for k,print(di[]) v in di.items():
# c = [4, 3, 6, 8, 4, 4, 5, 6, 9, 1]
# print(c)
# # c = set(c)
# # print(c)
# print(sorted(c))
# if 'a' in li:
#     print('a in list', li)
#     'CHCCHCCCH'
