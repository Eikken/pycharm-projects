#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数据处理VS_OS变窄.py    
@Time    :   2022/5/27 15:58  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   smooth ：x step = 5, get y mean value.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normHeat(heatMap):
    up = heatMap < -258.225
    down = heatMap > -257
    return up == down


def Os_flexible(x_, n_, line=1):
    if line == 2:
        # 第二条线，两条线的initial point不一样
        return n_ * (x_ + (1 / n_ - 1) * (-1.40742))
    return n_ * (x_ + (1 / n_ - 1) * (-1.36305))


def Vs_flexible(x_, n_, line=1):
    if line == 2:
        # 第二条线，两条线的initial point不一样
        return n_ * (x_ + (1 / n_ - 1) * 4.32754)
    return n_ * (x_ + (1 / n_ - 1) * 4.27726)


if __name__ == '__main__':
    fileOs = r'C:\Users\Celeste\Desktop\OS.csv'
    fileVs = r'C:\Users\Celeste\Desktop\VS.csv'
    Os_data = pd.read_csv(fileOs)
    Vs_data = pd.read_csv(fileVs)

    OsArr = np.array(Os_data).astype(np.float)
    VsArr = np.array(Vs_data).astype(np.float)

    OsLen = OsArr.shape[0]
    VsLen = VsArr.shape[0]
    step = 1000
    newOsArr = np.zeros((OsLen // step, 4))
    newVsArr = np.zeros((VsLen // step, 4))

    for i in range(OsLen//step):
        tmpOsArr = OsArr[i*step: step*(i+1), :]
        newOsArr[i, :] = np.mean(tmpOsArr, 0)

    for i in range(VsLen//step):
        tmpVsArr = VsArr[i*step: step*(i+1), :]
        newVsArr[i, :] = np.mean(tmpVsArr, 0)

    fig = plt.figure(figsize=(4, 7), dpi=200)
    ax1 = plt.subplot(211)
    plt.title('step=%d' % step)
    ax2 = plt.subplot(212)
    n = 0.5
    ax1.plot(Os_flexible(newOsArr[:, 0], n_=n), newOsArr[:, 1])
    ax1.plot(Os_flexible(newOsArr[:, 2], n_=n, line=2), newOsArr[:, 3])

    ax2.plot(Vs_flexible(newVsArr[:, 0], n_=n), newVsArr[:, 1])
    ax2.plot(Vs_flexible(newVsArr[:, 2], n_=n, line=2), newVsArr[:, 3])

    # n = 0.2
    # ax1.plot(Os_flexible(OsArr[:, 0], n_=n), OsArr[:, 1])
    # ax1.plot(Os_flexible(OsArr[:, 2], n_=n, line=2), OsArr[:, 3])
    #
    # ax2.plot(Vs_flexible(VsArr[:, 0], n_=n), VsArr[:, 1])
    # ax2.plot(Vs_flexible(VsArr[:, 2], n_=n, line=2), VsArr[:, 3])
    plt.show()
    print('finish')

