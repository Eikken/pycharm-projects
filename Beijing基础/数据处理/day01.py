import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 1.1, 0.1)
y = 2*x + 1
y1 = x**2
# print(y)
plt.figure(figsize=(8, 6))  # figsize 参数是长宽

plt.rcParams['font.sans-serif'] = 'SimHei'  # 改字体

plt.plot(x, y, markersize='10', marker='p', color='pink', linestyle='-', linewidth=1.0, markerFaceColor='g',
         markeredgecolor='m')
plt.plot(x,y1, markersize='10', marker='*', color='pink', linestyle='-', linewidth=1.0, markerFaceColor='g',
         markeredgecolor='m')
# marker= *\s\d\o\p 表示不同的标志, markersize 表示标志大小,linestyle是线的形式，-.长虚线，-?是短虚线，--连续虚线,:是密集点虚线
# linewith 是线的宽度,markerFaceColor是标志填充的颜色,markeredgecolor是标志的边颜色
plt.title('lines')
plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0]) #传列表

plt.xlabel('x轴')
plt.ylabel('y轴')
plt.show()
