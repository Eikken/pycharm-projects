#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   多项式拟合.py    
@Time    :   2021/4/27 18:19  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   https://blog.csdn.net/m0_38068229/article/details/105202554
把多项式拟合表示成矩阵相乘的形式
y(x,w)=w0 * x^0 + w1 * x^1 + w2 * x^2 +...+ wM * x^M
'''
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def regress(M, N, x, x_n, t_n, lam=0):
    print('M=%d, N=%d' % (M, N))
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    e = np.tile(order, [1, N])
    XT = np.power(x_n, e)
    X = np.transpose(XT)
    a = np.matmul(XT, X) + lam * np.identity(M + 1)  # X.T * X
    b = np.matmul(XT, t_n)  # X.T * T
    w = np.linalg.solve(a, b)  # aW = b =>  (X.T * X) * W = X.T * T
    print('W:', w)
    e2 = np.tile(order, [1, x.shape[0]])
    XT2 = np.power(x, e2)
    p = np.matmul(w, XT2)
    return p


if __name__ == '__main__':
    t1 = time.time()
    filename = r'../data/Size_30°.xls'
    cols = ['size', 'over_lap_area', 'over_lap_number', 'over_lap_ratio']
    df = pd.read_excel(filename)
    M = 9
    N = 365
    lamda_ = np.exp(0)
    x_data = df['size'][:N].values
    xx = np.linspace(15, 380, N)
    y_data = (df['over_lap_area'][:N].values / 100)
    pp = regress(M, N, xx, x_data, y_data)
    plt.scatter(x_data, y_data, 1)
    plt.plot(xx, pp, 'r', lw=1)
    plt.savefig('png/Multi_%d.png' % M)
    print('showed, saved png Multi = %d' % M)
    plt.show()
