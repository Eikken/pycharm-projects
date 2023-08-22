import pandas as pd
import numpy as np

file = pd.read_excel(r'C:\Users\Administrator\Desktop\meal_order_detail.xlsx')
data = file[file.columns]
# 出去NaN是255345个值
f1 = 0
for i in data.columns:
    f1+=1
    print(i)
print('删除前共',f1,'个标签\n')
for i in file:
    if file[i].describe().loc['count'] == 0 : # count空值进行删除
        # print(i,'is null') # 单列打印所有信息
        data.drop(labels=i,axis=1,inplace=True)
for i in file:
    first_value = file[i][0]
    flag = 0.0
    for j in file[i]:
        if j==first_value:
            flag += 1
    if file[i].describe().loc['count'] == flag and flag != 0 : # 每一行都相同的的值删除
        # print(i,'has the same value')
        data.drop(labels=i, axis=1, inplace=True)

f1 = 0
for i in data.columns:
    f1+=1
    print(i)
print('删除后共',f1,'个标签')

for i in data.columns:
    pass# print(data[i])