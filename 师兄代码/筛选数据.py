#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   筛选数据.py    
@Time    :   2021/8/18 16:40  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   提取指定格式的数据项
'''


def selfPrint(ct):
    if ct % 10 == 0:
        print(ct)
    else:
        print(ct, end=' ')


if __name__ == '__main__':
    filePath = r'C:\Users\Celeste\Desktop\reset\voutput.lammps'
    fileName = filePath.split('\\')[-1].split('.')[0] + '.txt'
    # 就是路径文件名改成.txt，当然也可以注释掉fileName, 自己定义fileName = '*.txt'
    fileData = open(filePath)
    line = fileData.readline()
    startLine = 'ATOMS id type vx vy vz'
    endLine = 'ITEM: TIMESTEP'
    count = 0
    while len(line) != 0:
        line = fileData.readline()
        if startLine in line:
            # tmpList = []
            l = fileData.readline()
            while endLine not in l:
                with open(fileName, 'a') as f:
                    f.write(l)
                    l = fileData.readline()
            count += 1
            selfPrint(count)  # 主要是为了看代码进行到哪了，注释掉可以不print
    fileData.close()
