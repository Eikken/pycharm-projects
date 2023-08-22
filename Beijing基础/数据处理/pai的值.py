#!/user/bin python
#coding=UTF-8
'''
@author  : Eikken
#@file   : readData.py
#@time   : 2019-09-16 23:13:34
'''
import pandas as pd
import numpy as np
import math

# file = pd.read_excel(r'C:\Users\Administrator\Desktop\附件1.xlsx')
# data = file[file.columns]
theta = 1.0
for i in range(4):
    print("PI=",end="")
    pi = (180/theta)*math.sin(theta)
    theta = theta/10
    print(pi)