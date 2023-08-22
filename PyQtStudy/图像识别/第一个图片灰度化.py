from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# 获取图片
def getimg():
    # return Image.open("../image/njtech.jpg")
    return Image.open("C:/Users/Celeste/Desktop/BaTiO-2.jpg")

# 显示图片
def showimg(img, isgray=False):
    plt.axis("off")
    if isgray == True:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    showimg(getimg(),isgray=False)
    # 变灰色
    im = getimg()
    im_gray = im.convert('L')
    showimg(im_gray, True)


    # im = getimg()
    # im_gray1 = im.convert('L')
    # im_gray1 = np.array(im_gray1)
    # avg_gray = np.average(im_gray1)
    # im_gray1 = np.where(im_gray1[..., :] < avg_gray, 0, 255)
    # showimg(Image.fromarray(im_gray1), True)

    # im = getimg()
    # im_gray = im.convert('L')
    # im3 = 255.0 * (im_gray / 255.0) ** 2
    # showimg(Image.fromarray(im3))

    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.title('灰度变换函数图像')
    # plt.xlabel('像素值')
    # plt.ylabel('变换后像素值')
    #
    # x1 = np.arange(0, 256)
    # y1 = np.arange(0, 256)
    #
    # f1, = plt.plot(x1, y1, '--')
    #
    # y2 = 255 - x1
    # f2, = plt.plot(x1, y2, 'y')
    #
    # y3 = (100.0 / 255) * x1 + 100
    # f3, = plt.plot(x1, y3, 'r:')
    #
    # y4 = 255.0 * (x1 / 255.0) ** 2
    # f4, = plt.plot(x1, y4, 'm--')
    # plt.legend((f1, f2, f3, f4), ('y=x', 'y=255-x', 'y=(100.0/255)*x+100', 'y=255.0*(x/255.0)**2'), loc='upper center')
    # plt.show()
