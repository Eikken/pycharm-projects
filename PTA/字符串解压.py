#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   字符串解压.py    
@Time    :   2022/10/22 16:54  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


if __name__ == '__main__':
    s = ''
    # line = input().split('-')

    line = 'ab-fzs-o-o-q'.split('-')

    for i in range(len(line)-1):
        start = line[i][-1]
        end = line[i+1][0]
        if len(line[i]) != 1:
            s += line[i][:len(line[i]) - 1]

        if start == end:
            s += ''

        elif start > end:

            for j in range(ord(start), ord(end), -1):
                s += chr(j)
        else:
            for j in range(ord(start), ord(end)):
                s += chr(j)
    s+=line[-1][-1]
    print(s)