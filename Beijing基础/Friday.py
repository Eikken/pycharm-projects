import requests
import re

from lxml import etree

response = requests.get("https://api.bilibili.com/x/v1/dm/list.so?oid=71008195")
content = response.content.decode("utf-8")

pattern = re.compile("<d.*?>(.*?)</d>")
result = pattern.findall(content)

for i in result:
    with open('弹幕.txt','a',encoding='UTF-8') as f:
        f.write(i+'\n')

# url = "http://www.shanbay.com/wordlist/110521/232414/?page=%d"
headers = {'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400' }
#
# flag = 0
# for page in range(1,4):
#     new_url = url%page  #重新组合所需要的URL地址
#     response = requests.get(url=new_url,headers=headers)
#     #用requests返回response结果
#     tree = etree.HTML(response.content.decode("UTF-8"))
#     #利用 etree返回网页的Xpath树
#
#     tr_list = tree.xpath('//table[@class="table table-bordered table-striped"]/tbody/tr')
#     # 找到主要要爬的Xpath地址
#     for i in tr_list:
#         word = i.xpath('./td/strong/text()')
#         # ./代表当前路径，获取当前路径下的文本用Xpath+/text()，
#         translator = i.xpath('./td[2]/text()')
#         #同上
#         print(word[0],translator[0])
#         #返回一个列表，打印每个列表的第一个元素即可
#         flag += 1
#
# print("共",flag,"个单词")



    # with open("扇贝第%d页.html"%page,"w",encoding="UTF-8") as f:
    #     f.write(response.content.decode("UTF_8"))
# response = requests.get("http://langlang2017.com/")
# res = response.content.decode("utf-8")
# #print(res)
# print("联系电话：",end='')
# pattern = re.compile("\d{11}")
# phone = pattern.findall(res)
# print(phone)
# pattern = re.compile("<div class=\"beian\">(.*?)<a.*?>(.*?)</a>")
# address = pattern.findall(res)
# print("备案：",end='')
# print(address[0][0],address[0][1])

#
# content = response.content.decode("UTF-8")
# #print(content)

with open("logo.png","wb") as f:
    f.write(response.content)

