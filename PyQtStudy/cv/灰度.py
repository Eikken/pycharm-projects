import cv2

image = cv2.imread("../image/njtech.jpg")
# print(type(image)) # <class 'numpy.ndarray'>
# # 先定义窗口，后显示图片
# cv2.namedWindow('njtech', cv2.WINDOW_NORMAL)
# cv2.imshow('njtech', image)
# cv2.waitKey(0)
# image[100, 100] = [255,255,255]# 行对应y，列对应x，所以其实是`img[y, x]`,修改像素值
# area = image[100:300, 115:188]
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('njtech', cv2.WINDOW_NORMAL)
cv2.imshow('njtech', img_gray)
cv2.waitKey(0)
# print(image.size)
# height, width, channels = image.shape # 彩色返回一个包含*行数（高度）、列数（宽度）和通道数*的元组，灰度图只返回行数和列数