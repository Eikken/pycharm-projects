import requests
from lxml import etree
import requests
import bs4
import re

link_url = "https://www.baidu.com/bh/dict/ydxx_8158835209873076610?tab=%E6%A6%82%E8%BF%B0&title=%E8%82%9D%E7%99%8C&contentid=ydxx_8158835209873076610&query=%E8%82%9D%E7%99%8C&sf_ref=dict_home&from=dicta"



headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
}

response = requests.get(url=link_url, headers=headers)

tree = etree.HTML(response.content.decode("UTF-8"))
rootxp = '//*[@id="richTextContainer"]/div' #根目录，大题目
xp = '//*[@id="richTextContainer"]/div[1]/div[1]'
xpgaishu = '//*[@id="richTextContainer"]/div[1]/div[2]/ul'
richang = '//*[@id="richTextContainer"]/div[7]/comment()'
x_path_list = tree.xpath(rootxp)
jieshao = '//*[@id="richTextContainer"]/div[1]/div[2]/ul'
jinfo = '//*[@id="richTextContainer"]/div[1]/div[2]/ul/li[4]'
for j in range(1,5):
    name = tree.xpath(jieshao)[0].xpath('./li['+str(j)+']/text()')
    print(name)

for i in range(0,7):
    name = x_path_list[i].xpath('div[1]/text()')
    print(name)
    xp_info = 'div[2]/div/div/p/text()'
    second = x_path_list[i].xpath(xp_info)
    print(second)
    wd = name[0] + '\n'
    for k in second:
        wd += k + '\n'
    with open('HCC.txt','a',encoding='utf-8') as f:
        f.write(wd)
        f.close()
print('爬完了')
# xpath_list = tree.xpath('/html/body/ul')
# name = xpath_list[num].xpath('./li/a/text()')
# href = xpath_list[num].xpath('./li/a/@href')
# print("name:", name, "\nURL:", href)