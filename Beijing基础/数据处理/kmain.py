import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from numpy import *


def randX():
    x1 = random.randint(10, 30)
    x2 = random.randint(10, 30)
    x3 = random.randint(10, 30)
    return x1, x2, x3


def randY():
    y1 = random.randint(100, 300)
    y2 = random.randint(100, 300)
    y3 = random.randint(100, 300)
    return y1, y2, y3


def draw():  # (width, height, dpi):
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 字体
    plt.rcParams['axes.unicode_minus'] = False  # 字符编码
    return plt.figure(figsize=(16, 9), dpi=100)  # 返回一个plt画布


def new_x(container):  # 后期转移类中心坐标的时候用得到取新的x, y坐标
    return int(np.sum(container[:, 0]) / len(container))


def new_y(container):
    return int(np.sum(container[:, 1]) / len(container))


if __name__ == '__main__':
    file = pd.read_csv(r'C:\Users\Administrator\Desktop\company.csv', encoding='gbk')
    arr = np.array(file[['平均消费周期（天）', '平均每次消费金额']])
    # for content in arr:
    #     print(content)

    x1, x2, x3 = randX()
    y1, y2, y3 = randY()
    # 获取三个随机数点。
    points = np.array([[x1, y1], [x2, y2], [x3, y3]])  # 随机points集合,三个点
    print('初始随机点坐标为：', points[0], points[1], points[2])
    pic = draw()
    pic_num = 1
    pic.add_subplot(2, 4, pic_num)  # 一个两行三列的画布
    plt.scatter(arr[:, 0], arr[:, 1])  # 原始数据散点图
    plt.scatter(points[(0, 1, 2), 0], points[(0, 1, 2), 1], marker='*', color='lightpink')  # 随机数散点，这是矩阵

    container = {
        's': [],
        'v': [],
        'c': []
    }  # 产生一个保存分类的容器

    while True:  # 死循环
        pic_num += 1  # next picture draw number
        pic.add_subplot(2, 4, pic_num)  # next picture draw,添加到画布
        s = []
        v = []
        c = []
        # 三个临时类别容器
        for i in range(len(arr)):
            distance = np.sum(arr[i, :] - points, axis=1) ** 2  # 我没开方哈，有问题回来找我，完全背离了欧式距离的思想哈哈哈
            # distance 是三个类点到某一point的距离，保存在一个列表表里
            minIndex = np.argmin(distance)  # 返回其中的一个最小距离
            if minIndex == 0:
                s.append(arr[i, :])
            elif minIndex == 1:
                v.append(arr[i, :])
            elif minIndex == 2:
                c.append(arr[i, :])
        container['s'] = np.array(s)
        container['v'] = np.array(v)
        container['c'] = np.array(c)  # 存到字典
        # 下面我们进行类点变化
        x1 = new_x(container['s'])  # 就是类中点的x,y坐标加和然后除以点的个数
        x2 = new_x(container['v'])
        x2 = new_x(container['c'])
        y1 = new_y(container['s'])
        y2 = new_y(container['v'])
        y3 = new_y(container['c'])
        new_points = np.array([[x1, y1], [x2, y2], [x3, y3]])
        # 开始绘图,我就想用三种颜色区分
        plt.scatter(container['s'][:, 0], container['s'][:, 1], color='cyan')
        plt.scatter(container['v'][:, 0], container['v'][:, 1], color='burlywood')
        plt.scatter(container['c'][:, 0], container['c'][:, 1], color='lightpink')
        plt.scatter(new_points[:, 0], new_points[:, 1], marker='*', color='k')
        plt.title('第%d次迭代' % (pic_num - 1))
        if (new_points == points).all():
            print(points)
            break
        else:
            points = new_points
    plt.show()

##########################################################################3
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def Drawing(width, high, dpi):
#     plt.rcParams['font.sans-serif'] = 'SimHei'  # 更改字体格式（仿宋）
#     plt.rcParams['axes.unicode_minus'] = False  # 正常显示符号
#     p = plt.figure(figsize=(width, high), dpi=dpi)
#     return p
#
#
# def get_class():
#     classt = []
#     print('请给定分类点：\n(建议取值：[30, 150], [10, 300], [20, 250])')
#     for i in range(3):
#         a, b = map(int, input("类%d：\n" % (i + 1)).split())
#         temp = np.array([a, b])
#         classt.append(temp)
#     classt = np.array(classt)
#     return classt
#
#
# if __name__ == '__main__':
#     data = pd.read_csv(r'C:\Users\Administrator\Desktop\company.csv', encoding='gbk')
#     customer = data[['平均消费周期（天）', '平均每次消费金额']]
#     customer = np.array(customer)
#     p1 = Drawing(16, 9, 100)
#     classt = get_class()
#     axi = 1
#     axt = p1.add_subplot(2, 4, axi)
#     plt.scatter(customer[:, 0], customer[:, 1])
#     plt.scatter(classt[(0, 1, 2), 0], classt[(0, 1, 2), 1], marker='x')
#     class_group = {
#         'class1': [],
#         'class2': [],
#         'class3': []
#     }
#     while True:
#         axi += 1
#         axt = p1.add_subplot(2, 4, axi)
#         class1 = []
#         class2 = []
#         class3 = []
#         for cut in range(len(customer)):
#             classify = np.sqrt((np.sum(customer[cut, :] - classt, axis=1)) ** 2)
#             class_result = np.argmin(classify) + 1
#             if class_result == 1:
#                 class1.append(customer[cut, :])
#             elif class_result == 2:
#                 class2.append(customer[cut, :])
#             elif class_result == 3:
#                 class3.append(customer[cut, :])
#         class_group['class1'] = np.array(class1)
#         class_group['class2'] = np.array(class2)
#         class_group['class3'] = np.array(class3)
#         x1 = int(np.sum(class_group['class1'][:, 0]) / len(class_group['class1']))
#         y1 = int(np.sum(class_group['class1'][:, 1]) / len(class_group['class1']))
#         x2 = int(np.sum(class_group['class2'][:, 0]) / len(class_group['class2']))
#         y2 = int(np.sum(class_group['class2'][:, 1]) / len(class_group['class2']))
#         x3 = int(np.sum(class_group['class3'][:, 0]) / len(class_group['class3']))
#         y3 = int(np.sum(class_group['class3'][:, 1]) / len(class_group['class3']))
#         new_classt = np.array([[x1, y1], [x2, y2], [x3, y3]])
#         plt.scatter(class_group['class1'][:, 0], class_group['class1'][:, 1])
#         plt.scatter(class_group['class2'][:, 0], class_group['class2'][:, 1])
#         plt.scatter(class_group['class3'][:, 0], class_group['class3'][:, 1])
#         plt.scatter(new_classt[:, 0], new_classt[:, 1], marker='x')
#         plt.title('第%d次迭代结果' % (axi-1))
#         if (new_classt == classt).all():
#             break
#         else:
#             classt = new_classt
#     plt.show()
