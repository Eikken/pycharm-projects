#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   first_pytorch.py    
@Time    :   2022/10/12 17:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import pymatgen

z = torch.rand(3, 4)

print('z', z)