#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   pa弹幕.py    
@Time    :   2023/4/21 15:41  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
from lxml import etree


def func_here(*args, **kwargs):
    pass


if __name__ == '__main__':
    # start here
    link_url = "http://www.court.gov.cn/paper/default/index/page/%d.html"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
    }

    for page in range(1, 5):

        new_link = link_url % page
        response = requests.get(url=new_link, headers=headers)
        tree = etree.HTML(response.content.decode("UTF-8"))
        xpath_list = tree.xpath('//*[@id="container"]/div/div[3]/div[2]/ul/li')
        # Xpath 分析
        # ul / li[1] / div[1] / ul / li[1]
        # ul/li[1]/div[1]/ul/li[1]/a
        # ul / li[1] / div[1] / ul / li[2] / div[1]
        # ul / li[1] / div[1] / ul / li[2] / div[2]
        for xp in xpath_list:
            name = xp.xpath("./div/ul/li/a/text()")
            num = xp.xpath("./div/ul/li[2]/div/text()")
            time = xp.xpath("./div/ul/li[2]/div[2]/text()")
            info = str(name[0]) + "  " + str(num[0]) + "  " + str(time[0])
            # print(info)
            f = open("法院文书.txt", 'a')
            f.write(info + "\n")

        print("写完了。")