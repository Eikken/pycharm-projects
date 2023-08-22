#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   爬nature.py    
@Time    :   2021/3/29 14:00  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import requests as req
from lxml import etree
import os
import re

link_url = 'https://www.nature.com/search?q=transition-metal%20dichalcogenide&order=relevance&journal=nature'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
}


def getPath():
    Path = os.getcwd()
    picPath = os.path.join(Path,'fileName')
    if not os.path.isdir(picPath):
        os.mkdir(picPath)
    picPath += '\ '
    return picPath
# //*[@id="content"]/div/div/div/div[2]/div[2]/section/ol


html = req.get(url=link_url, headers=headers).content.decode('UTF-8')
xml = etree.HTML(html)
xpList = xml.xpath('//*[@id="content"]/div/div/div/div[2]/div[2]/section/ol/li')
for i in xpList:
    #//*[@id="content"]/div/div/div/div[2]/div[2]/section/ol/li[1]/div/h2/a
    name = i.xpath('./div/h2/a/text()')
    print(name)