#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   统一改文件名.py    
@Time    :   2023/3/28 20:48
'''

import os
import shutil

from_path = r'D:\Celeste\PycharmProjects\png'
to_path = r'D:\Celeste\PycharmProject'
fileNames = os.listdir(path=from_path)
for f in fileNames:
    # new_name = f.replace('.', '').replace('0001', '')
    # os.rename(from_path+'/'+f, from_path+'/'+new_name)
    shutil.move(from_path+'/'+f, to_path)
