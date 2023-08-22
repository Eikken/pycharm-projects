#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   测试.py    
@Time    :   2022/12/1 15:10  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    # start here
    outcar = open('data/OUTCAR(1).phon', 'r')
    outcar.seek(0)  # just in case
    epsilon = []
    while True:
        line = outcar.readline()
        if not line:
            break
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
            outcar.readline()
            epsilon.append([float(x) for x in outcar.readline().split()])
            epsilon.append([float(x) for x in outcar.readline().split()])
            epsilon.append([float(x) for x in outcar.readline().split()])
    outcar.close()
    print('>' * 8, epsilon)

    try:
        print("-->using try outcar.out ")
        epsilon.append([float(x) for x in outcar.readline().split()])
        epsilon.append([float(x) for x in outcar.readline().split()])
        epsilon.append([float(x) for x in outcar.readline().split()])

    except:
        from lxml import etree
        print("-->using except vasprun.xml")
        doc = etree.parse('data/vasprun.xml')
        epsilon = [[float(i) for i in c.text.split()] for c in
                   doc.xpath("/modeling/calculation/varray")[3].getchildren()]

        print(epsilon)
