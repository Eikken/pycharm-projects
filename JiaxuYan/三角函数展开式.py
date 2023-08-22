#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   三角函数展开式.py    
@Time    :   2021/6/9 15:50  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   最起码在单个连续边界晶胞的距离上是稳定的，
            然而我们得到，第一个晶胞有距离为 a (a=键长)
            a的夹角余弦就是arrow distance.


'''
import numpy as np
import matplotlib.pyplot as plt


def cot(x_):
    # x 绝对值属于(0,pi)
    return 1 / x_ - x_ / 3 - x_ ** 3 / 45 - 2 * x_ ** 5 / 945


def triangleArea(a_, Theta):
    Theta = np.deg2rad(Theta)
    return 0.5 * a_ ** 2 * np.sin(Theta)


def CosineTheorem(a_, Theta):
    Theta = np.deg2rad(Theta)
    return (2.0 * a_ ** 2 * (1 - np.cos(Theta))) ** 0.5


def getHypotenuse(a_, b_):
    return (a_ ** 2 + b_ ** 2) ** 0.5


spanLength = [142, 284, 375, 512, 568, 619, 710, 751, 790, 863, 931,
              994, 1024, 1109, 1136, 1162, 1213, 1238, 1262, 1354]

cellLength = {'6.01': 1354.862355, '7.34': 1109.275439, '9.43': 863.9236077, '10.42': 1354.8623546323813,
              '11.64': 1213.4891841297963, '13.17': 619.0864237, '15.18': 931.3409687, '16.43': 994.1971635,
              '17.9': 790.7793624, '21.79': 375.771207, '24.43': 1162.550644, '26.01': 1262.3739541039336,
              '27.8': 512.0898359}

# angle = map(float, cellLength)
aa, l = [], []
for k, v in cellLength.items():
    l.append(v)
    aa.append(k)
l = np.array(l)
x = np.array(aa)
y = np.array(spanLength)
for i in spanLength:
    plt.axhline(i, linewidth=1, linestyle='--', color='grey')
plt.plot(x, l)
plt.scatter(x, l)
plt.xlabel('angle')
plt.ylabel('span')
plt.title('SuperCell edge length association')
plt.show()
# 245.99999996423668   122.99999998211834
a = 142.0281662
b = 142.0281662 * np.cos(np.deg2rad(30))
c = a * 0.5
d = 0.5 * CosineTheorem(a, 120)
H = 71.0140831
i = 1
print(i, getHypotenuse(9*a+H, 3*d))  # 1398.8152127536637
print(i, getHypotenuse(9*a+H, d))  # 1354
i += 1
print(i, getHypotenuse(8*a+H, 3*d))  # 1262.3739539204103
i += 1
print(i, getHypotenuse(8*a, 4*d))  # 1238
i += 1
print(i, getHypotenuse(8*a+H, d))  # 1213.4891839533805
i += 1
print(i, getHypotenuse(8*a, 2*d))  # 1162.550643889131
i += 1
print(i, 8*a)  # 1136
i += 1
print(i, getHypotenuse(7*a, 4*d))  # 1109
i += 1
print(i, getHypotenuse(7*a, 2*d))  # 1024.1796715884427
i += 1
print(i, 7*a)  # 994
i += 1
print(i, getHypotenuse(6*a+H, d))  # 931.3409685758472
i += 1
print(i, getHypotenuse(5*a+H, 3*d))  #863
i += 1
print(i, getHypotenuse(5*a+H, d))  # 790
i += 1
print(i, getHypotenuse(5*a, 2*d))  # 751
i += 1
print(i, 5*a)  # 710
i += 1
print(i, getHypotenuse(4*a, 2*d))  # 619
i += 1
print(i, 4*a)  # 568
i += 1
print(i, getHypotenuse(3*a+H, d))   # 512
# i += 1
# print(i, getHypotenuse(2*a+H, d))  # 375.77120693174953
i += 1
print(i, getHypotenuse(2*a, 2*d))  # 375.77120693174953
i += 1
print(i, 2*a)  # 284
i += 1
print(i, a) # 142
# for i in range(5): # 1 # 1 # 3 # 3 # 5 #                      9
#     print(i, end=' >> ')
#     B = (i+1) * 2 * b
#     print(getHypotenuse(a, B))
# print()
# for i in range(6): # 1 # 1 # 3 # 3 # 5 #                      9
#     print(i, end=' >> ')
#     B = b * (1 + 2 * i)
#     print(getHypotenuse(c, B))
# print()
# for i in range(9):  # 1 # 1 # 3 # 3 # 5 #                      9
#     if (i+1) % 3 == 0:
#         continue
#     else:
#         print(i, end=' >> ')
#         print((i+1)*a)

# for k in angle:
#     for i in range(4):
#         print(k, '>>', CosineTheorem(spanLength[i], np.deg2rad(k)))
#     print()
  