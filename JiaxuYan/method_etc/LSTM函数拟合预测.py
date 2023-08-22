#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   LSTM函数拟合预测.py
@Time    :   2021/4/27 10:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   https://blog.csdn.net/weixin_36009861/article/details/112537062
Dense(
inputs: 输入数据，2维tensor.
units: 该层的神经单元结点数。
activation: 激活函数.
use_bias: Boolean型，是否使用偏置项.
kernel_initializer: 卷积核的初始化器.
bias_initializer: 偏置项的初始化器，默认初始化为0.
kernel_regularizer: 卷积核化的正则化，可选.
bias_regularizer: 偏置项的正则化，可选.
activity_regularizer: 输出的正则化函数.
trainable: Boolean型，表明该层的参数是否参与训练。如果为真则变量加入到图集合中 GraphKeys.TRAINABLE_VARIABLES (see tf.Variable).
name: 层的名字.
reuse: Boolean型, 是否重复使用参数
)
'''

import math
import random
import time
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
import matplotlib.pyplot as plt
from numpy import concatenate

# from tensorflow.python.keras import layers, models
# from keras.optimizers import SGD
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def kerasModel(xtrain, ytrain, xtest):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xtrain, ytrain, epochs=50, batch_size=72)
    predicted = model.predict(xtest)
    predictedArea = sc.inverse_transform(predicted)
    drawFig(realArea, predictedArea, 1)


def drawFig(reaArea, predArea, lw=1):
    plt.style.use('fivethirtyeight')
    plt.plot(reaArea, lw=lw, color='black', label='Real area')
    plt.plot(predArea, lw=lw, color='green', label='Predicted area')
    plt.xlabel('size')
    plt.ylabel('area')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    t1 = time.time()
    filename = r'../data/Size_30°.xls'
    cols = ['size', 'over_lap_area', 'over_lap_number', 'over_lap_ratio']
    # df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-20')
    df = pd.read_excel(filename)
    dataSets_1 = df[['size', 'over_lap_area']]

    # # 一共380个数据，选择前340个作为训练，后40个作为验证预测结果
    # x_data = dataSets_1['size'][:330]
    # y_data = dataSets_1['over_lap_area'][:330] / 100
    # x_Test = dataSets_1['size'][330:]
    # y_Test = dataSets_1['over_lap_area'][330:] / 100
    sc = MinMaxScaler(feature_range=(0, 1))
    data21 = sc.fit_transform(df.iloc[10:340, 1:2].values/100)  # 330组数据
    # print(data21[:10])
    x_train, y_train = [], []
    for i in range(30, 330):
        x_train.append(data21[i - 30: i, 0])
        y_train.append(data21[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    realArea = df.iloc[330:360, 1:2].values/100
    inputs = df.iloc[306:, 1:2].values/100
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(30, 60):
        x_test.append(inputs[i - 30: i, 0])
    x_test = np.array(x_test, dtype='float64')
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    kerasModel(x_train, y_train, x_test)
    t2 = time.time()
    print('Finish, using time ', t2 - t1)
