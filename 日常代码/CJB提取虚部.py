#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   CJB提取虚部.py    
@Time    :   2022/7/12 18:09  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import numpy as np


def func_1(a_, b_, c_, d_):
    sum1 = -2**0.5*np.sin(np.conj(a_))*np.sin(2*np.conj(c_))*1j/2
    sum2 = 2**0.5*np.cos(np.conj(a_))
    pass


if __name__ == '__main__':
    pass