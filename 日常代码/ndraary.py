# !/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   ndraary.py    
@Time    :   2021/5/23 15:28  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   如何输出ndarray
'''
import numpy as np
import matplotlib.pyplot as plt

k = [121, 222, '124', '134']
l = [21, 22, '24', '34']
plt.figure(figsize=(6, 6), edgecolor='black')
plt.arrow(k[0], k[1], -l[0], l[1], length_includes_head=True, width=0.1, color='red')
plt.show()