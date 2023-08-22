#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   爬取CAS代码.py    
@Time    :   2022/4/25 20:31  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import requests
from lxml import etree


# xpath = '//*[@id="artContent"]/table[1]/tbody/tr/td[2]'

if __name__ == '__main__':
    link_url = 'http://www.360doc.com/content/19/0510/13/13328254_834789734.shtml'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
    }
    response = requests.get(url=link_url, headers=headers)
    # print(response.text.encode("ISO-8859-1").decode("utf-8"))
    tree = etree.HTML(response.content.decode("UTF-8"))
    rootXpath = '//*[@id="artContent"]/table[1]/tbody/tr/td[2]/code'
    # // *[ @ id = "artContent"] / table[1] / tbody / tr / td[2]
    x_path_list = tree.xpath(rootXpath)
    for xp in x_path_list:
        name = xp.xpath("./text()")
        if '\xa0' not in name[0]:
            print(name[0], end=" ")