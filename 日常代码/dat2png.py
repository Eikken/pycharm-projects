#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   dat2png.py
@Time    :   2022/3/31 15:24  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import os


def imageDecode(dat_dir, dat_file_name):
    dat_read = open(dat_dir, "rb")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    out = target_path + "\\" + dat_file_name.split('.')[0] + ".png"
    png_write = open(out, "wb")
    for now in dat_read:
        for nowByte in now:
            newByte = nowByte ^ xor_value
            png_write.write(bytes([newByte]))
    dat_read.close()
    png_write.close()


def findFile(dat_path):
    fsinfo = os.listdir(dat_path)
    allNum = len(fsinfo)
    count = 0
    for dat_file_name in fsinfo:
        temp_path = os.path.join(dat_path, dat_file_name)
        if not os.path.isdir(temp_path):
            count += 1
            print('共%d个文件，已完成%.2f %%' % (allNum, count * 100 / allNum))

            # print('文件路径: {}'.format(temp_path))
            imageDecode(temp_path, dat_file_name)
        else:
            pass


if __name__ == '__main__':
    # 修改dat文件的存放路径
    dat_path = r'C:\Users\Celeste\Desktop\wechat'

    # 修改转换成png图片后的存放路径
    target_path = r'C:\Users\Celeste\Desktop\wechat\dat2png'

    xor_value = 0x80  # 每个人不一样

    findFile(dat_path)

    print('finish')
