#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   批量修改标签图片名.py    
@Time    :   2022/2/21 17:59  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import os
from moviepy.editor import ImageSequenceClip

path = r'E:\表情\最爱'
# from PIL import Image
# import os.path
# def convertjpg(jpgfile,outdir,width=120,height=120):
#     img=Image.open(os.path.join(outdir, os.path.basename(jpgfile)))
#     try:
#         new_img = img.resize((width, height), Image.BILINEAR)
#         if new_img.mode == 'P':
#             new_img = new_img.convert("RGB")
#         if new_img.mode == 'RGBA':
#             new_img = new_img.convert("RGB")
#         new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
#     except Exception as e:
#         print(e)
# fileList = os.listdir(path)
# for jpgfile in fileList:
#     print(jpgfile)
#     convertjpg(jpgfile,path)
#


# 获取该目录下所有文件，存入列表中
fileList = os.listdir(path)


n = 0

for i in fileList:
    # oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
    # newname = path + os.sep + str(n + 1) + '.gif'
    # clip = ImageSequenceClip([oldname], fps=1)
    # clip.write_gif(newname)
    # # 设置旧文件名（就是路径+文件名）
    # oldname = path + os.sep + fileList[n]  # os.sep添加系统分隔符
    # # 设置新文件名
    oldname = path + os.sep + str(n + 1) + '.PNG'
    newname = path + os.sep + str(n + 1) + '.png'
    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    # print(oldname, '======>', newname)
    n += 1