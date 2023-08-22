import matplotlib.pyplot as plt
import numpy as np

rad = np.arange(0, np.pi*2, 0.1)
p1 = plt.figure(figsize=(8, 6), dpi=80)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 第一个子图

ax1 = p1.add_subplot(2, 2, 1)  # 两两列第一个
plt.scatter(rad, np.sin(rad))  # 对应 x, y scatter 是散点图
plt.plot(rad, np.cos(rad))

plt.legend(['y=sin(x)', 'y=cos(x)'])  # 先画哪个先标哪个，先画后标
plt.title('正余弦曲线')
# plt.show() 不能提前show，会导致图像不完整

ax2 = p1.add_subplot(2, 2, 2)  # 第二个
plt.plot(rad, np.cos(rad))

ax3 = p1.add_subplot(2, 2, 3)  # 第三个
plt.plot(rad, np.cos(rad))

ax4 = p1.add_subplot(2, 2, 4)  # 第四个
plt.plot(rad, np.cos(rad))

plt.show()
