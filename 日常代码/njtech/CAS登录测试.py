#!/usr/bin python
# -*- encoding: utf-8 -*-
"""
@Author  :   Celeste Young
@File    :   CAS登录测试.py
@Time    :   2022/4/16 21:12
@E-mail  :   iamwxyoung@qq.com
@Tips    :   整个实现流程，需要请求两次：
                第一次请求，requests.get.refer_url，
                    如果成功，得到重定向页面url，并且带有cas service，基本都是成功的。

                第二次请求，带有 lt_value, execute、eventId参数去post信息,
                    如果成功，返回成功页面，使用session请求其他子级页面的访问。
                    如果失败，失去cookie认证，返回403 bad request.
            https://i.njtech.edu.cn/app/api/ipcheck
            {"ip":"124.235.234.95","status":false}

            可请求API
            南工新闻：https://i.njtech.edu.cn/papi/messages?role=postgraduate&type=youth

            应用中心：https://i.njtech.edu.cn/papi/apps?role=postgraduate&limit=500

            不可请求API
            个人信息
            https://i.njtech.edu.cn/app/profile/data?accountKey=XGH&accountValue=202061122133&uri=open_api/customization/mvyktuser/list
"""
import requests
from lxml import etree
import http.cookiejar as cookielib
from sqlalchemy import true


class casService(object):
    def __init__(self, svr_session):
        self.cas_url = 'https://u.njtech.edu.cn/cas/login'
        self.refer_url = 'https://u.njtech.edu.cn'
        self.redirect_url = 'https://i.njtech.edu.cn'
        self.index_url = 'https://i.njtech.edu.cn/index.html'
        self.profile_url = 'https://i.njtech.edu.cn/app/profile/data?accountKey=XGH&accountValue=202061122133&uri=open_api/customization/mvyktuser/list'
        self.svr_session = svr_session
        self.session = requests.Session()
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
                        "Accept": "text/html, application/xhtml+xml, application/xml; q=0.9, */*; q=0.8",
                        "Accept-Language": "zh_CN",
                        "Connection": "keep-alive",        }


if __name__ == '__main__':

    caS = casService(requests.session())

    cas_response = caS.session.get(caS.cas_url, headers=caS.headers, allow_redirects=False)
    if cas_response.status_code == 200:  # request HTML with no cas_service page
        print(caS.cas_url, ' 200 OK')
        print('> '*10)
        login_html = etree.HTML(cas_response.text)
        form_action = login_html.xpath('//*[@id="fm1"]/@action')  # return list; len()=1
        # form_action = ['/cas/login?service=https%3A%2F%2Fu.njtech.edu.cn%2Foauth2%2Fauthorize%3Fclient_id%3DOe7wtp9CAMW0FVygUasZ%26response_type%3Dcode%26state%3Dnjtech%26s%3Df682b396da8eb53db80bb072f5745232']
        execution_value = login_html.xpath('//*[@id="fm1"]/div/div[1]/div[4]/input[2]/@value')
        lt_value = login_html.xpath('//*[@id="fm1"]/div/div[1]/div[4]/input[1]/@value')
        event_value = login_html.xpath('//*[@id="fm1"]/div/div[1]/div[4]/input[3]/@value')
        auth_data = {'_eventId': event_value, 'lt': lt_value,
                     'submit': 'login', 'execution': execution_value,
                     'username': '202061122133', 'password': '586947abc'}

        newCasUrl = caS.refer_url + form_action[0]  # 拼接新的service_url
        auth_session = caS.session
        auth_session.cookies = cookielib.LWPCookieJar(filename='i_cookies.txt')
        auth_response1 = auth_session.post(newCasUrl, data=auth_data, headers=caS.headers, allow_redirects=False)
        # 登录成功, 需要session带有记录跳转 i.njtech.edu.cn

        auth_cookies = requests.utils.dict_from_cookiejar(auth_response1.cookies)
        # success_html = etree.HTML(auth_response1.text)
        # success_data = login_html.xpath('//*[@id="msg"]/h2/text()')
        # 获取session cookie中的cas TGT参数
        if auth_response1.status_code == 302:
            url_with_ticket = caS.refer_url + auth_response1.headers["Location"]
            cookie_session = requests.session()
            confirm_response = cookie_session.get(url=url_with_ticket, headers=caS.headers, allow_redirects=True)
            # url_with_ticket = caS.profile_url
            # success_response = cookie_session.post(url=url_with_ticket, headers=caS.headers, allow_redirects=True)
            print(confirm_response.text)
            if confirm_response.status_code == 200:
                print('login successful')
                # print(confirm_response.text)
            else:
                print('login failed')
        else:
            print("auth failed")
    else:
        print("cas cookie still valid")

    print("finish")
