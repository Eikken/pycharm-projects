# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns



class GraModel():
    '''灰色关联度分析模型'''

    def __init__(self, inputData, p=0.5, standard=True):
        '''
        初始化参数
        inputData：输入矩阵，纵轴为属性名，第一列为母序列
        p：分辨系数，范围0~1，一般取0.5，越小，关联系数间差异越大，区分能力越强
        standard：是否需要标准化
        '''
        self.inputData = np.array(inputData)
        self.p = p
        self.standard = standard
        # 标准化
        self.standarOpt()
        # 建模
        self.buildModel()

    def standarOpt(self):
        '''标准化输入数据'''
        if not self.standard:
            return None
        self.scaler = StandardScaler().fit(self.inputData)
        self.inputData = self.scaler.transform(self.inputData)

    def buildModel(self):
        # 第一列为母列，与其他列求绝对差
        momCol = self.inputData[:, 0]
        sonCol = self.inputData[:, 0:]
        for col in range(sonCol.shape[1]):
            sonCol[:, col] = abs(sonCol[:, col] - momCol)
        # 求两级最小差和最大差
        minMin = sonCol.min()
        maxMax = sonCol.max()
        # 计算关联系数矩阵
        cors = (minMin + self.p * maxMax) / (sonCol + self.p * maxMax)
        # 求平均综合关联度
        meanCors = cors.mean(axis=0)
        self.result = {'cors': {'value': cors, 'desc': '关联系数矩阵'}, 'meanCors': {'value': meanCors, 'desc': '平均综合关联系数'}}


# if __name__ == "__main__":
#     # 路径目录
#     curDir = os.path.dirname(os.path.abspath(__file__))  # 当前目录
#     baseDir = os.path.dirname(curDir)  # 根目录
#     staticDir = os.path.join(baseDir, 'Static')  # 静态文件目录
#     resultDir = os.path.join(baseDir, 'Result')  # 结果文件目录
#     # 读数
#     data = pd.read_excel('灰度表1.xlsx')
#     columns = ['线路价格（不含税）', '总里程', '业务类型', '需求类型1',
#                '需求类型2', '是否续签', '车辆长度', '车辆吨位', '打包类型',
#                '运输等级', '需求紧急程度', '计划卸货等待时长',
#                '计划运输时长（分钟）', '线路总成本']
#
#     data = data[columns]
#     data = np.array(data).T
#     # 建模
#     model = GraModel(data, standard=True)
#     print(model.result)
#     # 用来正常显示中文标签
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     # 用来正常显示负号
#     plt.rcParams['axes.unicode_minus'] = False
#     # 可视化矩阵
#     plt.clf()
#     plt.figure(figsize=(8, 12))
#     sns.heatmap(meanCors.reshape(1, -1), square=True, annot=True, cbar=False,
#                 vmax=1.0,
#                 linewidths=0.1, cmap='viridis')
#     plt.yticks([0, ], ['quality'])
#     plt.xticks(np.arange(0.5, 12.5, 1), columns, rotation=90)
#     plt.title('指标关联度矩阵')
#     plt.savefig(resultDir + '/指标关联度可视化矩阵.png', dpi=100, bbox_inches='tight')