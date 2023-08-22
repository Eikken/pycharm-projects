import requests
from lxml import etree

link_url = "https://book.douban.com/subject_search?search_text=python&cat=1001"
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
}

response = requests.get(url=link_url,headers=headers)
# 请求
tree = etree.HTML(response.content.decode("UTF-8"))
# Xpath树
xpath_list = tree.xpath('//*[@item="root"]/div/div[2]/div/div/div')
#//*[@id="root"]/div/div[2]/div[1]/div[1]
# name xpath ://*[@id="root"]/div/div[2]/div[1]/div[1]  /div[1]/div/div/div[1]/a
# name xpath ://*[@id="root"]/div/div[2]/div[1]/div[1]/div[1]
# name xpath ://*[@id="root"]/div/div[2]/div[1]/div[1]/div[2]
# name xpath ://*[@id="root"]/div/div[2]/div[1]/div[1]/div[2] /div/div/div[1]/a
#author xpath://*[@id="root"]/div/div[2]/div[1]/div[1]/div[1]  /div/div/div[3]
#score xpath://*[@id="root"]/div/div[2]/div[1]/div[1]/div[1]  /div/div/div[2]/span[2]
#img xpath : //*[@id="root"]/div/div[2]/div[1]/div[1]/div[1]  /div/a/img
print(xpath_list)
for xp in xpath_list:
    name = xp.xpath("./div/div/div/a/text()")
    author = xp.xpath("./div/div/div[3]/text")
    score = xp.xpath("/div/div/div[2]/span[2]")
    print(name,author,score)

