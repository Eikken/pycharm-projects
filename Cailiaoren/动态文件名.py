#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   动态文件名.py    
@Time    :   2021/3/3 17:40  
@Tips    :   https://www.runoob.com/python3/python3-os-file-methods.html
'''

import os,glob,sys,shutil

allList = os.listdir(os.getcwd())
dirList = []
fileList = []
for i in allList:
   if not os.path.isfile(i): # 是否是文件
       dirList.append(i)
   else:
       fileList.append(i)
for j in dirList: # 对所有文件夹进行迭代
    for i in fileList:
        if '动态文件名.py' not in i:#排除自己的脚本
            shutil.copy(i,os.path.join(os.getcwd(),j))
            # 将文件复制到j文件夹中
    os.chdir(os.path.join(os.getcwd(),j))        # 切换工作目录
    os.system('qsub vasp.pub')
    os.chdir('..')
# print(os.listdir(os.getcwd()))
# print(glob.glob('cla*.py'))

# # os.remove('nuw_mulu')
# os.removedirs('nuw_mulu')
# os.mkdir('nuw_mulu')
# os.chdir('nuw_mulu')
# shutil.copy('class1.py','./')
# print(os.listdir(os.getcwd()))
# file1 = open(sys.argv[1],'r')
