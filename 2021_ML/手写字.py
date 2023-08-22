#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   手写字.py    
@Time    :   2021/2/15 21:24  
@Tips    :   手写字和图片读取
'''

# # ===========手写体数据===========
from sklearn.datasets import load_digits,load_sample_image
import matplotlib.pyplot as plt # 画图工具
# digits = load_digits()
# data=digits.data
# print(data.shape)
# plt.matshow(digits.images[15])  # 矩阵像素点的样式显示
# # plt.imshow(digits.images[3])  # 图片渐变的样式显示3
# # plt.gray()   # 图片显示为灰度模式

img=load_sample_image('flower.jpg')   # 加载sk自带的花朵图案
plt.imshow(img)
plt.show()