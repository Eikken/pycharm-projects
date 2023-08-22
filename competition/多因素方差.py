from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Imputer
import pandas as pd
import numpy as np

dataSet = pd.read_excel('灰度表1.xlsx')
columns = ['交易成功时长','线路总成本','总里程', '业务类型',
           '需求类型2', '是否续签', '车辆长度',
           '打包类型','运输等级','标的展示策略',
            ]
# ]
# columns = ['线路价格（不含税）','总里程','线路总成本','需求紧急程度']

col = ["a","b","c","d","e",
       "f","g","h","i","j",
       ]

data = dataSet.loc[:, columns]

data.columns = col
formula = "a~ b + c + d + e + f + g + h + i + j  "
# formula = '线路价格（不含税）~ 总里程 + 业务类型 + 需求类型1 + 需求类型2 + 是否续签 + 车辆长度 + 车辆吨位 + 打包类型 + 运输等级 + 需求紧急程度 + 计划卸货等待时长 + 计划运输时长 + 线路总成本'
anova_results = anova_lm(ols(formula, data).fit())
print(anova_results)
