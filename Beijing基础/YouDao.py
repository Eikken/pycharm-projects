from urllib import request,parse
import time
import hashlib

def getSign(data): # MD5加密算法
    md5 = hashlib.md5()
    md5.update(data.encode("utf-8"))
    return md5.hexdigest()

print("***有道翻译在线***")
url = "http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule"
headers = {
'Accept':'application/json, text/javascript, */*; q=0.01',
#'Accept-Encoding':'gzip, deflate',
'Accept-Language':'zh-CN,zh;q=0.9',
'Connection':'keep-alive',
'Content-Length':'250', #当前值250位最小匹配
'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
'Cookie':'OUTFOX_SEARCH_USER_ID=1211305132@10.168.8.64; JSESSIONID=aaaQKdQWAU4GQa_r5EEHw; OUTFOX_SEARCH_USER_ID_NCOO=755099156.7567027; ___rl__test__cookies=1547775013402',
'Host':'fanyi.youdao.com',
'Origin':'http://fanyi.youdao.com',
'Referer':'http://fanyi.youdao.com/',
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400',
'X-Requested-With':'XMLHttpRequest'
}
while True:
    md_5= hashlib.md5()
    info = input("请输入单词：")
    if info == "000":
        break
    salt = str(int(time.time()*10000))
    ts = str(int(time.time()*1000))
    sign_data = "fanyideskweb" + info + salt + "p09@Bn{h02_BIEe]$P^nG"
    data = {
    "i" : info,
    "from":"AUTO",
    "to": "AUTO",
    "smartresult": "dict",
    "client": "fanyideskweb",
    "salt": salt,
    "sign": getSign(sign_data),
    "ts": ts,
    "bv": "3e2596225753e80305665b73042cf4bf",
    "doctype": "json",
    "version": "2.1",
    "keyfrom": "fanyi.web",
    "action": "FY_BY_REALTIME",
    "typoResult": "false"
    }
    data = parse.urlencode(data) #编译一下
    req = request.Request(url=url,headers=headers,data=bytes(data,encoding="utf-8")) #包装头部
    response = request.urlopen(req).read().decode("utf-8") #返回响应信息
    print(response)
    import json

    obj = json.loads(response)
    for rows in obj["smartResult"]["entries"]:
        print(rows)

    #"{"translateResult":[[{"tgt":"hello","src":"你好"}]],"errorCode":0,"type":"zh-CHS2en"}"