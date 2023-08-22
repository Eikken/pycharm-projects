#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class2.py    
@Time    :   2021/1/19 20:15  
@Tips    :   文件类操作
'''

# open() 函数用于打开文件，并创建一个文件对象
# close()函数用于关闭对象 有借有还，再借不难
# file object=open(name[ mode,encoding ])
# e name 为要打开的文件名，字符串
# e mode 为打开文件的模式，字符串，默认为只读，即 'r‘
# g encoding 为读取或写入文件的编码方式。
#模式 名称 可读 可写 如文件不存在 文件指针位置
# ‘r’ read 是 否 报错 开头
# ‘r+’ read + write 是 是 报错 开头
# ‘w' write 否 是 创建 开头
# ‘w+' write + read 是 是 创建 开头
# ’a' write(append) 否 是 创建 结尾
# ’a+' append + read 是 是 创建 结尾
'''
file = open('data/XDATCAR','r',encoding='utf8')
content = file.read()
print(content) # this text's real data
print(file) # address of <_io.TextIOWrapper name='data/XDATCAR' mode='r' encoding='utf-8'>

# Using "with open('data/XDATCAR','r',encoding='utf8') as file: file.read()" will colse file automatically
# We offen use "read()"(with pointer remove untill last) "readline()"(with pointer remove to next line)  "readlines()"
# (with pointer remove untill last and show as a list of remain lines)  to read file's data.

# string.strip ("\ \r r\ \ n").strip('\ \ n')
print(file.read ()) #把file里面的内容全部读入
file.seek( (0) ) #文件指针在最后了，移到最前面，准备重新读
print(file.readline ()) #只读第一行，文件指针到了第二行
print(file.readlines()) #从第二行读到最后，保存为列表
file.close() # remember close to avoid unknown error

# 关于排序
seqList = [2, 6, 3, 7, 5, 4, 9]
print(sorted(seqList)) # 原列表不改变，返回一个新的已排序的列表
seqList.sort() # 原列表改变
print(seqList)
seqList.sort(reverse=True) # 从大到小排序
print(seqList)
seqList.sort(reverse=False) # 从小到大排序 Warning：注意Python的深浅复制！仓库和钥匙的关系
print(seqList)
# 字符串排序默认按照ASCii码排序
'''

# 读取二维石墨烯CIF文件, 目标：读取晶格大小，原子的标签和坐标分量
def stripFormatting(string):
    return string.strip('\r\n').strip('\n')

file = open('data/bigQuasi.cif','r',encoding='utf8')

jingGeInfo = []
dictJingGe = {}
atomInfo = []
currentIndex = 0
tmpIndex = 0

content = file.readlines()

for i in content:
    if '_cell_length' in i: #判断晶格信息是不是在这一行
        dictJingGe[stripFormatting(i).split()[0]] = float(stripFormatting(i).split()[1])
       # jingGeInfo.append(float(stripFormatting(i).split()[1]))
    if '_atom_site' in i:
        atomInfo.append(stripFormatting(i)) #存储_atom_site 字段用来确定label 和xyz分量所有原子信息中的第几列
        tmpIndex = currentIndex
    currentIndex += 1
# print(dictJingGe)
# print(atomInfo)
# 我们要获取atomSite的第几列的信息，返回list.index
# labelIndex, xIndex, yIndex, zIndex = 0, 0, 0, 0
indexList = []
for i in range(len(atomInfo)):
    if '_type' in atomInfo[i]:
        indexList.append(i) # labelIndex = i
    if '_x' in atomInfo[i]:
        indexList.append(i) # xIndex = i
    if '_y' in atomInfo[i]:
        indexList.append(i) # yIndex = i
    if '_z' in atomInfo[i]:
        indexList.append(i) # zIndex = i
atom_label_xyz = []  # ['O', 0.88722, 0.96062, 0.0]
atomDict = {} # {'O':0, 'Ba':0, 'Ti':0}
for i in range(tmpIndex+1,len(content)):
    if len(content[i].split()) == len(atomInfo):
        thisLine = stripFormatting(content[i])
        label = thisLine.split()[indexList[0]]
        if label in atomDict.keys():
            atomDict[label] += 1
        else:
            atomDict[label] = 1
        x = float(thisLine.split()[indexList[1]])
        y = float(thisLine.split()[indexList[2]])
        z = float(thisLine.split()[indexList[3]])
        atom_label_xyz.append([label,x,y,z])

# for i in atom_label_xyz:
#     print(i)
# print(atomDict)
# print('length of atoms:',len(atom_label_xyz))
