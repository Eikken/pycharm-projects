#!/usr/bin python
# -*- encoding: utf-8 -*-
'''
@Author  :   Celeste Young
@File    :   研究生管理系统登录.py    
@Time    :   2022/4/29 12:19  
@E-mail  :   iamwxyoung@qq.com
@Tips    :   
'''
import json

import ddddocr
import os
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import requests


def localTestVerify():
    pass
    # req = requests.get("http://yjsxt.njtech.edu.cn/home/verificationcode?codetype=stucode")
    # # print()
    # ocr = ddddocr.DdddOcr(show_ad=False)
    # img_bytes = req.content
    # res = ocr.classification(img_bytes)  # 识别率 99% !
    # with open(f'img_%s.gif' % str(res), 'wb') as f:
    #     f.write(img_bytes)
    # print('finish >> ', res)


def local_JSEncrypt(jsObj):
    key = '''-----BEGIN PUBLIC KEY-----
MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDsASvbyQk835FZUrtswSdsikHC
Cbi7qmhZY9spnHOWNvG2wYlSMb3ugVgxGEKQw010Xu86do6ZmUc0WOu0jywd52ew
gIwRG00PPuRJl7UgWOaTy0anF13r/5nwtpbit2z/BhHWYojLS8jFmb7MnNXpvCnF
faQPeXngYAZS1BxJfwIDAQAB
-----END PUBLIC KEY-----
'''  # 注意上述key的格式
    rsakey = RSA.importKey(key)
    cipher = Cipher_pkcs1_v1_5.new(rsakey)  # 生成对象
    cipher_text = base64.b64encode(cipher.encrypt(jsObj.encode(encoding="utf-8")))  # 对传递进来的用户名或密码字符串加密
    value = cipher_text.decode('utf8')  # 将加密获取到的bytes类型密文解码成str类型
    return value


if __name__ == '__main__':
    req = requests.get("http://yjsxt.njtech.edu.cn/home/verificationcode?codetype=stucode")

    # print()
    ocr = ddddocr.DdddOcr(show_ad=False)
    img_bytes = req.content
    res = ocr.classification(img_bytes)  # 识别率 99% !

    post_url = 'http://yjsxt.njtech.edu.cn/home/stulogin/stulogin_do'
    info = {
        'UserId': '202061122133',
        'Password': '586947abc',
        'VeriCode': res,
    }
    headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'
    }
    system_url = 'http://yjsxt.njtech.edu.cn/home/stulogin'
    jsonStr = local_JSEncrypt(json.dumps(info))
    data = {'json': jsonStr}
    login = requests.post(url=post_url, data=data, headers=headers)
    print(login.text)
