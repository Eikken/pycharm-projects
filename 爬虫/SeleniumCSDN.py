#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   SeleniumCSDN.py    
@Time    :   2021/7/8 16:20  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

# -*- coding:utf-8 -*-
import os
import time
from selenium import webdriver # 从selenium导入webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import json
import time

#引入chromedriver.exe
chromedriver="C:/Users/lex/AppData/Local/Google/Chrome/Application/chromedriver.exe"
os.environ["webdriver.chrome.driver"] = chromedriver
browser = webdriver.Chrome(chromedriver)
#设置浏览器需要打开的url
url = "https://passport.csdn.net/login?code=public"
browser.get(url)
browser.find_element_by_link_text("账号密码登录").click()
browser.find_element_by_id("all").clear()
browser.find_element_by_id("all").send_keys("你的邮箱地址")
time.sleep(1)
browser.find_element_by_id("password-number").clear()
browser.find_element_by_id("password-number").send_keys("你的登录密码")
time.sleep(1)
browser.find_element_by_css_selector("[class='btn btn-primary']").click()