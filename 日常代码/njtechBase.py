#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   njtechBase.py    
@Time    :   2022/4/16 13:04  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''


class NjtechBase(object):
    def __init__(self):
        self.handler = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/88.0.4324.150 Safari/537.36', }

        self.njtech_url = 'https://i.njtech.edu.cn/app/profile/data?accountKey=XGH&accountValue=%s&uri=open_api' \
                          '/customization/mvyktuser/list '

        self.Authorization_url = 'https://u.njtech.edu.cn/oauth2/authorize?client_id=Oe7wtp9CAMW0FVygUasZ&response_type=code&state' \
                                 '=njtech&redirect_uri=https://i.njtech.edu.cn/authorize'

        self.Data = {"username": "202061122133", "password": "586947abc"}

        BaseUrl = 'https://u.njtech.edu.cn'
        actionUrl2 = '/cas/login?service=https%3A%2F%2Fu.njtech.edu.cn%2Foauth2%2F'
        actionUrl3 = 'authorize%3Fclient_id%3DOe7wtp9CAMW0FVygUasZ%26response_type%3Dcode%26state%3D'
        actionUrl4 = 'njtech%26s%3Df682b396da8eb53db80bb072f5745232'
        self.actionUrl = BaseUrl + actionUrl2 + actionUrl3 + actionUrl4
