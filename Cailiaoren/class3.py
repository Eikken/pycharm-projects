#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class3.py    
@Time    :   2021/1/19 22:22  
@Tips    :   正则表达式的应用
            * matching unit character or characters for 0 or n times
            + matching the previous subexpression at least 1 times
            ? matching the previous child expression only 0 or 1 times
            *? non-greedy model, matches as few subexpressions as possible
            {n} match the previous subexpression n times
            {n,m} match the previous subexpression at least n times but no more than m times.
            Python re.findall()的可选参数
            re.I 匹配时对大小写不敏感
            re.M 跨行匹配，把$匹配换行符\n
            re.S 跨行匹配，使 . 也包含换行符
            使用writer写入文件。
'''
# ^username:(.*?); passwd:(.*?)$
import re

# def zhengZe(string):
#     pattern = '^username:(.*?); passwd:(.*?)$'
#     return re.findall()

# with open('data/in.in') as f:
#     content = f.readlines()
# pattern = '^username:(.*?);passwd:(.*?)$'
# for i in content:
#     print(re.findall(pattern,i)) # 返回一个元组，即正则表达式的内容
#
# p = re.findall('(\w{1})bc','abcAbc',re.I) # ['a', 'A']

with open('data/CuO111_2.xsd','r') as file:
    contents = file.read()
pattern1 = '[ABC]Vector=\"(.*?)\"'
abcv = re.findall(pattern1,contents)
vectors = []
for i in abcv:
    vectors.append([float(j) for j in i.split(',')]) # nice work of 'for' looping
pattern2 = '<Atom3d(.*?)XYZ=\"(.*?)\".*?Components=\"(.*?)\"'
abcAtom = re.findall(pattern2,contents)
atoms = []
elements = [i[0] for i in abcAtom]
for i in abcAtom:
    if 'RestrictedProperties' in i[0]:
        restrict = True
    else:
        restrict = False
    atoms.append((i[-1],i[1],restrict))
ele = []
for i in range(len(elements)):
    tmplist = elements[i].split()
    for j in range(len(tmplist)):
        if 'Name' in tmplist[j]:
            string = tmplist[j].split('"')[1]
            ele.append(string) # 切割出元素名称
        # ele.append(re.findall('Name=\"(.*?)\"',e.split()[index]))
eleDict = {} # {'CU': 10, 'O': 10}
for i in ele:
    if i not in eleDict.keys():
        eleDict[i] = 1
    else:
        eleDict[i] += 1
with open('data/POSCAR.bak','w') as writer:
    writer.write('POSCAR221205 by Celeste\n')
    writer.write('1.0\n')
    for i in vectors:
        writer.write('%7.3f %7.3f %7.3f\n'%(i[0],i[1],i[2]))
    for k in eleDict.keys():
        writer.write(str(k)+' ')
    writer.write('\n')
    for v in eleDict.values():
        writer.write(str(v)+' ')
    writer.write('\n')
    writer.write("Selective\nDirect\n")  #使VASP 读入原子固定信息 也就是原子坐标后的 T F #分数坐标的方式
    for i in atoms:
        if i[2] == False:
            suffix = 'T T T'
        else:
            suffix = 'F F F'
        tmp = ([float(j) for j in i[1].split(',')])
        writer.write('%7.3f %7.3f %7.3f %s\n'%(tmp[0],tmp[1],tmp[2],suffix))
    print('finish')