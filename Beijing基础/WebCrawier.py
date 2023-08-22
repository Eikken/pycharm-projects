from urllib import request,parse
print("*****百度翻译小程序*****")
url = "https://fanyi.baidu.com/sug"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400' }
while True:
    info = input("请输入单词：")
    if info == "退出":
        break
    data = {"kw":info}
    data = parse.urlencode(data) #转换

    req = request.Request(url=url,headers=headers,data=bytes(data,encoding="utf-8"))
    response = request.urlopen(req).read().decode("utf-8")

    import json
    obj = json.loads(response)
    #obj["data"] #这是一个字典
    for rows in obj["data"]:
        print(rows["k"],":",rows["v"])
print("您已退出百度翻译小程序")
# url = "http://tieba.baidu.com/f?"
# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400' }
# #kw = input("请输入页码：")
# for page in range(1,5):
#     data = {
#         "kw":"java",
#         "ie":"utf-8",
#         "pn": (page-1)*50
#     }
#     info = parse.urlencode(data)
#     new_url = url + info
#     req = request.Request(url=new_url,headers=headers)
#     response = request.urlopen(req).read().decode("utf-8")
#     print("正在写入java第%d页"%page)
#     with open("java第%d页.html"%page,"w",encoding="utf-8") as f:
#         f.write(response)

# new_url = url + "&%s"%info
# req = request.Request(url=new_url,headers=headers)
# response = request.urlopen(req).read().decode("utf-8")
# with open(kw+".html","w",encoding="utf-8") as f:
#     f.write(response)
# print(info)


# url = "http://tieba.baidu.com/f?ie=utf-8&fr=search"#"http://www.baidu.com"
# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400' }
#
# kw = input("请输入关键字：")
# new_url = url + "&kw=%s"%kw
#
# req = request.Request(url = new_url,headers = headers)
# response = request.urlopen(req).read().decode("utf-8")
# with open(kw+".html","w",encoding="utf-8") as f:
#     f.write(response)
# 创建URL，创建headers，URL中去掉&kw部分自己输入并且自己组成一个new_url
# 请求 = request.Request(url = ?,headers = ?)
# 响应 response = request.urlopen(请求).read().decode("utf-8")
# 写入文件 with open(文件名，"w",encoding = "utf-8") as f:
    # f.write(response) 写响应


# response = request.urlopen(url)
# content = response.read()
# with open("tieba.html","w",encoding="utf-8") as f:
#     f.write(content.decode('utf-8'))
#
# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400' }
# req = request.Request(url = url,headers = headers)
#
# response = request.urlopen(req).read().decode("utf-8")
# print(response)