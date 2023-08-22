#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   产生石墨烯标准.py    
@Time    :   2022/12/7 9:46  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

from pybinding.repository import graphene
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt

pb.pltutils.use_style()


def rectangle(width, height):
    x0 = width / 2
    y0 = height / 2
    return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])


if __name__ == '__main__':
    # start here
    model = pb.Model(
        graphene.monolayer(),
        rectangle(width=1.2, height=1.2)
    )
    # plt.figure(figsize=(10, 8), dpi=300)
    model.plot(site={'radius': 0.015})
    plt.xticks([])
    plt.yticks([])
    plt.show()