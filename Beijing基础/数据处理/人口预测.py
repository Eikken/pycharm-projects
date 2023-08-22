
# 城乡男女比例表
# 绘制各年份男女人口（每年的男女放在一起）及城乡人口直方图
# 男女人口比例及城乡人口饼图
#

import matplotlib.pyplot as plt
import numpy as np

d = np.load(r'C:\Users\Administrator\Desktop\populations.npz')
# print(d)

data = d['data.npy']
info = d['feature_names.npy']  # 表头
# print(info)
# print(data)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# ##################################################################################
# child = plt.figure(figsize=(8, 6), dpi=100)
# x = np.arange(0,20,1)
# labelsYear = data[-3:-23:-1, 0]
# child.add_subplot(2, 1, 1)
# plt.bar(x, data[-3:-23:-1, 2], width=0.25, color='pink')
# plt.plot(x, data[-3:-23:-1, 2], marker='*', markerFaceColor='pink' )
# plt.bar(x+0.25, data[-3:-23:-1, 3], width=0.25, color='cyan')
# plt.plot(x+0.25, data[-3:-23:-1, 3], marker='o', markerFaceColor='cyan' )
# plt.legend(['男','女','男','女'])
# plt.xticks(x, labelsYear, rotation=-45)
# plt.xlabel('年份')
# plt.ylabel('人口数量')
# plt.title('年份男女总人口人口直方图')
#
# child.add_subplot(2, 1, 2)
# plt.bar(x, data[-3:-23:-1, 2], width=0.5, bottom=data[-3:-23:-1, 3], color=['cyan'])  # bottom 标注当前的底部在哪，下面尚未填充
# plt.bar(x, data[-3:-23:-1, 3], width=0.5, color=['pink'])
# plt.plot(x, data[-3:-23:-1,1 ],marker="*", markerFaceColor='black')
# plt.xticks(x, labelsYear, rotation=-45)
# plt.xlabel('年份')
# plt.ylabel('人口数量')
# plt.savefig('年份男女人口直方图')
# plt.show()
# ##################################################################################
#
# plt.figure(figsize=(8, 6))
# x = np.arange(0,20,1)
# labelsYear = data[-3:-23:-1, 0]
# plt.bar(x, data[-3:-23:-1, 4], width=0.25, color='darkcyan')
# plt.plot(x, data[-3:-23:-1, 4], marker='*', markerFaceColor='darkcyan' )
# plt.bar(x+0.25, data[-3:-23:-1, 5], width=0.25, color='burlywood')
# plt.plot(x+0.25, data[-3:-23:-1, 5], marker='o', markerFaceColor='burlywood' )
# # plt.plot(x, data[-3:-23:-1,1 ],marker="*", markerFaceColor='black')
# plt.legend(['城镇','乡村','城镇','乡村'])
#
# plt.xticks(x, labelsYear, rotation=-45)
#
# plt.xlabel('年份')
# plt.ylabel('人口数量')
# plt.title('年份城乡人口直方图')
#
# plt.savefig('年份城乡人口直方图')
# plt.show()
# ################################################################################
# 大饼图
#
# explode = [0.01,0.01]
# labelBG = ['男','女']
# child = plt.figure(figsize=(8, 7), dpi=100)
# plt.title('年份男女人口饼状图\n')
# for i in range(20):
#     child.add_subplot(4, 5, i+1)
#     plt.pie(data[-3-i , 2:4], explode=explode, labels=labelBG, autopct='%1.1f%%',colors=['pink','cyan'])
#     plt.title(str(1996+i)+'年')
# plt.savefig('年份男女人口饼状图')
# plt.show()

# ################################################################################

explode = [0.01,0.01]
labelBG = ['城市','乡村']
child = plt.figure(figsize=(8, 7), dpi=100)
plt.title('年份城乡人口饼状图\n')
for i in range(20):
    child.add_subplot(4, 5, i+1)
    plt.pie(data[-3-i , 4:], explode=explode, labels=labelBG, autopct='%1.1f%%',colors=['darkcyan','burlywood'])
    plt.title(str(1996+i)+'年')
plt.savefig('年份城乡人口饼状图')
plt.show()

print("画完了")