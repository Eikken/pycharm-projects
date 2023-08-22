#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   函数测试_1.py    
@Time    :   2021/4/21 19:57  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np
from matplotlib import pyplot as plt
import math

x = np.arange(0, 1000, 1)
lamda = 2 ** 0.5 - 1
lamda2 = 2 - 3 ** 0.5
lamda3 = 7 - 4 * (3 ** 0.5)
lamda4 = 1.5
sai = x
h = lamda
y1 = np.sin(2*np.pi*lamda4*x)
y2 = 2*np.arcsin(y1)
y3 = 1*np.cos(sai) ** 2 + 1 * np.cos(sai)
y4 = np.sin(x)
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(1, 1, 1)
#, projection='polar')
# ax1.scatter(x, y1, 0.2, color='blue')
# ax1 = fig.add_subplot(2, 1, 2)
sai = np.pi / 2
print('V(pi/2)=', 0.5*np.cos(sai) ** 2 + h * np.cos(sai))
# ax1.plot(x, y4, 0.5, color='green')
ax1.scatter(x, y4, 0.5, color='red')
plt.savefig('png/fig_5.png', dpi=600)
plt.show()

# import networkx as nx
# G = nx.Graph(directed=False)
# G.add_node((0,0))
#
# for n in range(4):
#     for (q,r) in list(G.nodes()):
#         G.add_edge((q,r),(q,r-1))
#         G.add_edge((q,r),(q-1,r))
#         G.add_edge((q,r),(q-1,r+1))
#         G.add_edge((q,r),(q,r+1))
#         G.add_edge((q,r),(q+1,r-1))
#         G.add_edge((q,r),(q+1,r))
#
# pos = nx.spectral_layout(G)
# nx.draw(G,pos,alpha=.75)
#
# import pylab as plt
# plt.axis('equal')
# plt.show()