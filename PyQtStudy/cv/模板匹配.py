import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def getPic():
    fileList = os.listdir(r"D:/edu/njtech/tezhengyangpin/test")  # 获取文件夹下所有文件名返回list
    return fileList

picList = getPic()
for p in picList[20:]:
    # print(p) #(1320,0).jpg
    img = cv2.imread("D:/edu/njtech/tezhengyangpin/test/"+str(p),0)
    img2 = img.copy()
    tmp = cv2.imread("D:/edu/njtech/tezhengyangpin/target1.jpg",0)
    w, h = tmp.shape[::-1]
    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
    #相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
    #归一化相关系数匹配CV_TM_CCOEFF_NORMED
    for meth in methods:
        img = img2.copy()
        method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,tmp,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(p,res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, 255, 10)
        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(p)
        plt.show()