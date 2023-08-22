import numpy as np

# numpy 是矩阵和数组运算
arr = np.array([[1, 2, 3], [4, 5, 6]])  # 二维数组
# print(arr)

np.savez('arr_1', arr)
data = np.load(r'C:\Users\Administrator\Desktop\国民经济核算季度数据.npz')  #读npz文件
columns =  data['columns']
values = data['values']
# for i in columns:
#     print(i,end="  ")
# print(values)
# print(values[::4, 6:9:2])  #逗号之前表示第几行，逗号之后表示第几列
# [start:end:step,起始列:结束列:步长]

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
p1 = plt.figure(figsize=(8, 8), dpi=80)
x = range(12)
labels = columns[3:]
ax1 = p1.add_subplot(2, 1, 1)
plt.bar(x, values[68, 3:], width=0.5)
plt.xticks(x, labels,rotation=90)
plt.title('2017各产业')
plt.xlabel('产业')
plt.ylabel('增加值（亿元）')

plt.show()
# print(labels)

# for i in values:
#     print(i,end=' ')
