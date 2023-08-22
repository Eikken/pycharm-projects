import matplotlib.pyplot as plt
import numpy as np

labels = ['第一产业', '第二产业', '第三产业']
explode = [0.1, 0.01, 0.01]


plt.figure(figsize=(6, 6))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
data = np.load(r'C:\Users\Administrator\Desktop\国民经济核算季度数据.npz')  #读npz文件
columns = data['columns']
values = data['values']

plt.pie(values[-1, 3:6], explode=explode, labels=labels, autopct='%1.1f%%', colors=['r', 'g', 'b'])

plt.title("2017年国民生产总值")

plt.savefig('2017年国民生产总值')  # 先保存后显示
plt.show()
# explode 表示每一个饼尖到圆心之间的距离是 explode
# autopct = %结构值%
