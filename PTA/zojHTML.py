# !/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   zojHTML.py
@Time    :   2022/9/9 14:39
@E-mail  :   iamwxyoung@qq.com
@Tips    :    学会使用Python的EOF操作
'''


if __name__ == '__main__':
    import sys

    data = ''
    while True:
        line = sys.stdin.readline()
        data += line
        if not line:
            break

    strTmp = ''
    strPrint = ''
    lastStr = ''
    enter = False
    line = False
    for i in data.split():

        strPrint = strTmp

        if '<br>' in i:
            enter = True
        if '<hr>' in i:
            line = True

        if strTmp == '':
            strTmp = i
        else:
            strTmp += ' ' + i

        if enter:
            enter = False
            # print(strPrint)
            lastStr += strPrint + '\n'
            strTmp = ''
        elif line:
            line = False
            if strPrint != '':
                # print(strPrint)
                lastStr += strPrint + '\n'
            # print('-'*80)
            lastStr += '-'*80 + '\n'
            strTmp = ''

        elif len(strTmp) >= 80:
            # print(strPrint)
            lastStr += strPrint + '\n'
            strTmp = i

    # print(strTmp)
    lastStr += strTmp
    print(lastStr)
