#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   selinum测试登录njtech.py    
@Time    :   2022/4/3 10:46  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
from bs4 import BeautifulSoup
from selenium import webdriver
import time
from lxml import etree
import requests
import requests.utils
import http.cookiejar as cookielib
from 日常代码.njtechBase import NjtechBase

njtechSess = requests.Session()
njtechSess.cookies = cookielib.LWPCookieJar(filename="njtechCookies.txt")
userAgent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"
header = {
    "Referer": "https://i.njtech.edu.cn/",
    'User-Agent': userAgent,
}


if __name__ == '__main__':
    nb = NjtechBase()
    njtech_url = nb.njtech_url
    Authorization_url = nb.Authorization_url
    handler = nb.handler
    loginData = nb.Data
    actionUrl = nb.actionUrl

    # XGH = '202061122133'
    print("开始模拟登录")
    postUrl = "https://u.njtech.edu.cn/cas/login"
    # 使用session直接post请求
    response = njtechSess.get(postUrl, data=loginData, headers=header)
    # 无论是否登录成功，状态码一般都是 statusCode = 200
    print(response.status_code, postUrl)
    # print(responseRes.text)
    njtechSess.cookies.save('njtechCookies.txt')
    # personalUrl = 'https://i.njtech.edu.cn/personal/center'
    # responseRes = njtechSess.get(personalUrl, headers=header, allow_redirects=False)
    # print(responseRes.status_code, personalUrl)