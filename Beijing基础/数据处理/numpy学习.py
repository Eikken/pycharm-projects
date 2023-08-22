
import numpy as np

## 创建数组的方式

s = 'wsrhelle : 9.099809, sx  yxy'
print(s[::-1])
# arr_1 = np.array([1, 2, 3])
# arr_2 = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 8, 9]]) # 列表中的列表
# print(arr_1.ndim)
# print(arr_2.ndim) # ndim表示数组维度
# print(arr_2[2, 1:])
# print(arr_2[:3, 2]) # 切片查看数据
# print(arr_2[-1::-2, -1::-2])
#
# print(arr_2[(0, 2), (0, 2)]) # 表示arr_2[行,列] 值
# print(arr_2[(0, 1, 2), (2, 2, 0)])  #行列对应匹配获取数据

#
# # 数组.ndim是返回数组的维度
#
# arr_3 = np.arange(12)
# # print(arr_3.reshape(3, 4))
# # print(arr_3.reshape(3, -1))
#
# arr_4 = arr_3.reshape(3, -1)
# # reshape是一维转多维数组
# # print(arr_4.ravel())
# # ravel方法表示展平数组，多维到一维
# # print(arr_4)
# arr_5 = np.array([[0,1,2,1], [1,2,2,1], [0,1,0,1]])
# # print(arr_5)
#
# # print(arr_4+arr_5) # 直接加就完事了
# arr_6 = [1,2,3,4]
# x = np.array([1,3,5])
# y = np.array([1,4,3])
# print(x<y)

# print(arr_5+arr_6) # 数组的广播机制

# print(np.sum(arr_6,axis=1)) # 数组求和
# print(np.max(arr_6)) # 求最大值
# print(np.min(arr_6)) # 求最小值

#
# d = np.load(r'C:\Users\Administrator\Desktop\populations.npz')
# # print(d)
#
# data = d['data']
# info = d['feature_names']  # 表头

# print(data,info)
#
# for i in np.sum(data[0:20, 2:4], axis=1): # axis = 1 是按行，axis = 0 是按列
#     print(i)  # 按行相加，加完后print
#
# boy = data[0:20, 2]
# girl = data[0:20, 3]
# # 先读数据，然后数组性质相加
# print((boy+girl)==data[0:20,1])
#

