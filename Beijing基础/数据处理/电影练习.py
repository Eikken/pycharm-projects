# 机器学习，监督和无监督学习，根据是否有标签分析，
# 使用train模型和test模型进行训练和学习
# KNN算法，K-nearest neighbor，K最近邻算法
# 数据转化成坐标
#
import pandas as pd
import math
import sortedcollections
import numpy as np
###################################### 老师讲的
d = pd.read_excel(r'电影分类数据.xlsx')
# print(d.columns) # 打印表头

colums = np.array(d.columns) # 转成数组
# print(colums) #打印数组
test = colums[-3:]
# print(test) #提取要评测样本数据
train = d[['搞笑镜头', '拥抱镜头', '打斗镜头', '电影类型']] #提取谁写入谁,提取训练样本
# knn 算法：
train_data = train[['搞笑镜头', '拥抱镜头', '打斗镜头']] #pandas 数据提取必须写表头名，和numpy不一样
sort_index = np.argsort(np.sqrt(np.sum((train_data - test)**2, axis=1)))[:5] #argsort 是按索引index排序
# print(sort_index)
value = train['电影类型'] # [sort_index].mode().values# .mode() 众数
print(value)

###################################### 我自己做的
# d = pd.read_excel(r'C:\Users\Administrator\Desktop\电影分类数据.xlsx',encoding='gbk')
# # print(d)
# # print(d['搞笑镜头'][0])
# # print(d['拥抱镜头'][1])
# # print(d['打斗镜头'][2])
# arr = [0 for i in range(12)]
# for i in range(12):
#     arr[i] = '%.2f'%math.sqrt((d['搞笑镜头'][i]-23)**2 + (d['拥抱镜头'][i]-3)**2 + (d['打斗镜头'][i]-17)**2)
#
#
# s = d['电影名称'][0]
# dic_name= { s:arr[0] } # 构造一个字典
# dic_type= { s:d['电影类型'][0] }
# for i in range(1,12):
#     dic_name[d['电影名称'][i]] = arr[i]
#     dic_type[d['电影名称'][i]] = d['电影类型'][i]
#
# # for i in dic_type.items():
# #     print(i)
# # np.argsort()是按位置返回的排序
#
#
# new_dic = sorted(dic_name.items(), key=lambda x:x[1], reverse=False)
# ke = 5
#
# # for i in range(k):
# #     print(new_dic[i])
# count = 0
# dic = {'喜剧片':[0],'动作片':[0], '爱情片':[0]} #count 电影类型的count
#
# for i in range(ke):
#     for k,v in dic.items():
#         if dic_type[new_dic[i][0]]==k:
#             v[0] = v[0]+1
# max = 0
# key = ''
# for k,v in dic.items():
#     if max < v[0]:
#         key = k
#         max = v[0]
#
# print('在key='+str(ke)+'的情况下，你找的大概是个:'+key)
#     # print(dic_type[new_dic[i][0]]) # type
#     # print(new_dic[i][0]) # name


# ('我的特工爷爷', '17.49')
# ('美人鱼', '18.55')
# ('功夫熊猫3', '21.47')
# ('宝贝当家', '23.43')
# ('澳门风云3', '32.14')
# ('新步步惊心', '34.44')
# ('夜孔雀', '39.66')
# ('代理情人', '40.57')
# ('伦敦陷落', '43.42')
# ('谍影重重', '43.87')
# ('奔爱', '47.69')
# ('叶问3', '52.01')





