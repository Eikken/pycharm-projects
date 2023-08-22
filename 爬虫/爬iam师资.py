#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   爬iam师资.py    
@Time    :   2021/5/20 13:43  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
from lxml import etree
import requests
import bs4
import re



link_url = "http://iam.njtech.edu.cn/info/1033/3474.htm"

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
}

response = requests.get(url=link_url, headers=headers)
# print(response.text.encode("ISO-8859-1").decode("utf-8"))
tree = etree.HTML(response.content.decode("UTF-8"))
rootXpath = '/html/body/table[2]/tbody/tr/td[3]/table[2]/tbody/tr/td/div/div'
# # '/html/body/table[2]/tbody/tr/td[3]/table[2]/tbody/tr/td/div/div/p[2]'
rootXpath = '/html/body/table[2]/tbody/tr/td[3]/table[2]/tbody/tr/td/div/div/p[2]/span/strong'#[1]'
x_path_list = tree.xpath(rootXpath)
print(x_path_list)
# # for i in range(1, 5):
# print(x_path_list)