import pandas as pd

# csv 类似 txt 文件
order = pd.read_csv(r'C:\Users\Administrator\Desktop\meal_order_info.csv',encoding='gbk') # encoding 声明编码
# print(order['number_consumers'][:10]) # 取前十行
# print(order.columns) # 查看表头,即列索引
# print(order.index) # 查看行数
# print(order.values) # 取所有值

# print(order.T) # 把表横向查看

form =  pd.read_excel(r'C:\Users\Administrator\Desktop\users.xlsx')

print(form)

# 有标签叫监督学习，没标签叫无监督学习

