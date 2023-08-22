#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   class4.py    
@Time    :   2021/3/3 11:15  
@Tips    :   关于numpy和 matplotlib
            numpy.loadtxt(filename,dtype,comments='#',delimiter=None
            ,skiprows=0, usecols=None)
            filename 是文本文件名
            dtype 是读入数据的格式
            comments=‘#’, ‘#’开头的注释行就会跳过
            delimiter 行内数据的分隔符，空格的话numpy会自动判断
            skiprows 跳过文件的前（0）行
            usecols 只使用文本文件的(m-n)列数据
'''
import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('data/BAND-REFORMATTED.dat')
with open('data/KLABELS', 'r') as reader:
    all_ = reader.readlines()
klabels = []
kpath_crd = []
for i in all_[1:]:
    if len(i.split()) < 2:
        break
    else:
        klabels.append(i.split()[0])
        kpath_crd.append(float(i.split()[1]))
for i in range(1, data.shape[1]):
    plt.plot(data[:, 0], data[:, i])
plt.ylim(-15, 10)  # -15eV to 10 eV
plt.xticks(kpath_crd, klabels)
plt.axhline(y=0, xmin=0, xmax=1, linestyle='--', color='grey')  # 水平辅助线
for i in kpath_crd:
    plt.axvline(x=i, ymin=0, ymax=1, linestyle='--', color='grey')
plt.xlabel('Kpath')
plt.ylabel('Energy E-Efermi')
plt.title('Band Relation')
plt.savefig('png/bandpower.png', dpi=300)
plt.show()

# nparr1 = np.array([1, 2, 3, 4])
# # 可以下标访问，也可以使用.tolist()方法吧nparr转为list
# # print(type(nparr2.tolist()))
#
# nparr2 = np.array([1, 2, 3, 4,'5'])
# # nparr2 全变成字符串了，通过.astype(np.int) 转化成np的int型或者float型
# # print(nparr2)
# # print(nparr2.astype(np.int))
#
# # 多维列表
# nparr3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# nparr4 = np.array(9)
# nparr5 = np.arange(1,10).reshape(-1,3) # (3,-1)三行的数据，(-1,3)转化成三列的数据 ;是np.arange()方法
# # print(nparr5)
# nparr6 = np.zeros((3,3))
# nparr6[1] = [1, 2, 3]
# # print(nparr6)
# nparr_5 = nparr5[0:2, 1:] # 第0到1行，第1到最后一列，:不包括后面尾数
# # 对于np.arr拷贝要用.copy()方法。要不然只是视图截取
# # print(nparr5[0:2, 1:]>4) #返回布尔型索引[[False False],[ True  True]]
# # nparr5[nparr5>5] = -1 # 注意数组维度
# # print(nparr5)
# 1 / nparr5 # 取倒数
# np.dot(nparr5*nparr5) # 点乘法
# np.linalg.inv(nparr5) # 求nparr5的逆矩阵
# np.loadtxt() # 读取文本文件
#
# def func(x):
#     return 5 + 2*x + 3*x**2-0.5*x**3
# def fund(x):
#     return 200-10*x
# def mymin(x):
#     return min(func(x),fund(x))
#
# x = np.linspace(-10,10,100) # -4到4分100个区间
# y = x*x
#
# y1 = list(map(mymin,x))
#
# plt.title('%d points in total'%(len(y)))
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x,func(x),color='red')
# plt.plot(x,fund(x),linestyle="--",color='blue')
# # plt.scatter(x,y)
# plt.savefig('png/class4.png')#
# plt.show()
