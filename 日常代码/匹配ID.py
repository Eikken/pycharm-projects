#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   匹配ID.py    
@Time    :   2021/11/17 11:01  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import networkx as nx
import matplotlib.pyplot as plt

# 添加边
H = nx.path_graph(8)
F = nx.Graph()  # 创建无向图
F.add_edge(11, 12)  # 一次添加一条边

# 等价于
e = (13, 14)  # e是一个元组
F.add_edge(*e)  # 这是python中解包裹的过程

F.add_edges_from([(1, 2), (1, 3)])  # 通过添加list来添加多条边

# 通过添加任何ebunch来添加边
F.add_edges_from(H.edges())  # 不能写作F.add_edges_from(H)

nx.draw(F, with_labels=True)
plt.show()
print('图中所有的边', F.edges())

print('图中边的个数', F.number_of_edges())
