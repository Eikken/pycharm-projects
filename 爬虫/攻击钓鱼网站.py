#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   攻击钓鱼网站.py    
@Time    :   2022/12/6 10:23  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   http://cbhsfo.com/ms.html?email=18@qq.com
    time1 = time.time()
    time2 = time.time()
    print('>> Finished, use time %d s' % (time2 - time1))
'''
import random

import requests
from lxml import etree


def random_qq(*args, **kwargs):
    rand_qq = str(random.random()).split('.')
    qq = 'admin'
    if int(rand_qq[1][0]) < 3:
        qq = int(rand_qq[1][:10])
    else:
        qq = int(rand_qq[1][:random.randint(6, 9)])
    return qq


if __name__ == '__main__':
    rq = random_qq()
    link_url = 'https://urban3d.co/wordfence/img/index.php?email=%d@qq.com' % rq
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
    }
    response = requests.get(url=link_url, headers=headers)
    print(response.text.encode("ISO-8859-1").decode("utf-8"))
    tree = etree.HTML(response.content.decode("UTF-8"))
    passwd_xpath = '/html/body/table/tbody/tr[4]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/table/tbody/tr[4]/td/table/tbody/tr/td[3]/input'
    # rootXpath = '//*[@id="artContent"]/table[1]/tbody/tr/td[2]/code'
    auth_data = {'login': '%d@qq.com' % rq, 'passwd': rq}
    submit_button = '/html/body/table/tbody/tr[4]/td/table/tbody/tr[4]/td/table/tbody/tr/td[2]/table/tbody/tr[6]/td/table/tbody/tr/td[3]/div/input'
    