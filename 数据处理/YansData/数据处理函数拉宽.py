#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   数据处理函数拉宽.py    
@Time    :   2022/5/27 14:18  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   f(x) =
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flexibleFunction(x_, n_):
    return 1 / n_ * (x_ + (n_ - 1)*(-1.37542))


if __name__ == '__main__':

    filePath = r'C:\Users\Celeste\Desktop\Sheet1.csv'
    data = pd.read_csv(filePath)
    valueData = data != '--'
    dataSet = np.array(data[valueData]).astype(np.float)
    newDataSet = np.zeros((49999, 4))
    new_x = flexibleFunction(dataSet[:, 0], 0.333)
    newDataSet[:, 0] = [i for i in new_x]
    newDataSet[:, 1] = dataSet[:, 1]
    newDataSet[:, 2] = dataSet[:, 2]
    newDataSet[:, 3] = dataSet[:, 3]
    # print(newDataSet[:10])
    df = pd.DataFrame(newDataSet) # 数据转Excel
    df.to_excel('dataPackage/flex_data_0.333.xlsx', index=False, header=False)
    # fig = plt.figure(figsize=(6, 4), dpi=200)
    # # # ax1 = plt.subplot(211)
    # # # ax2 = plt.subplot(212)
    # # #
    # # # ax1.plot(dataSet[:, 0], dataSet[:, 1])
    # # # ax1.plot(dataSet[:, 2], dataSet[:, 3])
    # #
    # plt.plot(newDataSet[:, 0], newDataSet[:, 1])
    # plt.plot(newDataSet[:, 2], newDataSet[:, 3])
    #
    # plt.show()
    print('finish')
