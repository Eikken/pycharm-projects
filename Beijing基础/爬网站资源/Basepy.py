#!/user/bin python
#coding=UTF-8
'''
@author  : Eikken
#@file   : Basepy.py
#@time   : 2019-06-05 15:45:44
'''

import requests as req
from lxml import etree
import os
import re

link_url = 'https://www.137fb.com/vod/html16/index_%d.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3704.400 QQBrowser/10.4.3587.400'
}

def getPath():
    Path = os.getcwd()
    picPath = os.path.join(Path,'pic')
    if not os.path.isdir(picPath):
        os.mkdir(picPath)
    picPath += '\ '
    return picPath

def Download(url,name):
    data = req.get(url).content
    print(name)
    with open(getPath() + '%s'%name,'wb') as f :
        f.write(data)

def DownloadImage(pStart,pEnd):
    pEnd += 1
    for i in range(pStart,pEnd):
        newLink = link_url%i
        if i == 1:
            newLink = 'https://www.137fb.com/vod/html16/index.html'
        threadList = []
        html = req.get(url=newLink, headers=headers).content.decode('UTF-8')
        xml = etree.HTML(html)
        # //*[@id="content"]/li[7]  /div[1]/h5/a
        # //*[@id="content"]/li[1]   /a
        XpList = xml.xpath('//*[@id="content"]/li')
        print('第',i,'页正在下载中.',end='\n')
        for xp in XpList:
            name = xp.xpath('./div[1]/h5/a/text()')
            picHref = xp.xpath('./a/@style') # style 返回的是一个列表
            r = r'https:.+?\.jpg'
            jpgList = re.findall(r,picHref[0]) # 找到列表了需要的链接，返回列表
            jpgLink = jpgList[0]
            jpgHtml = req.get(url=jpgLink, headers=headers)
            # print(type(jpgHtml.content))
            # jpgXml = etree.HTML(jpgHtml)
            # /html/body/img
            # jpgXp = jpgXml.xpath('/html/body/img')
            # print(jpgXp)
            with open(getPath()+ '%s.jpg'%name[0],'wb') as f:
                f.write(jpgHtml.content)

def Deletepic():
    Path = os.getcwd()
    dePath = os.path.join(Path,'pic\\')
    files = os.listdir(dePath)
    # os.removedirs 是删除目录，os.remove是删除文件
    for f in files:
        os.remove(dePath+f)
    print('图片已删除！')
if __name__ == '__main__':
    html = req.get(url='https://www.137fb.com/vod/html16/index.html', headers=headers).content.decode('UTF-8')
    r = r'<title>.*?</title>'
    title = re.findall(r,html)[0]
    r = r'[\u4e00-\u9fa5]{4}'
    title = re.findall(r,title)[0]
    print('当前页面：',title)
    pStart = int(input('请输入起始下载页码：'))
    pEnd = int(input('请输入结束下载页码：'))
    DownloadImage(pStart,pEnd)
    # Deletepic()


# response = requests.get(url=iam_bsh_url, headers=headers)
# tree = etree.HTML(response.content.decode())
# xpath_list = tree.xpath('/html/body/table[2]/tr/td[3]/table[2]/tr/td/div')
# for xp in xpath_list:
#     info = xp.xpath("./p/a/text()")
#     print(info[0].split(' ')[0])
# njtech_url = 'https://u.njtech.edu.cn/cas/login?service=' \
#       'https%3A%2F%2Fu.njtech.edu.cn%2Foauth2%2Fauthorize%3Fclient_id%3DOe7wtp9CAMW0FVygUasZ%26' \
#       'response_type%3Dcode%26state%3Dnjtech%26s%3Df682b396da8eb53db80bb072f5745232'


# browser = webdriver.Firefox()  # Get local session of firefox
# browser.get(iam_bsh_url)  # Load page
# # assert "Yahoo!" in browser.title
# elem1 = browser.find_element_by_name("username")  # Find the query box
# elem2 = browser.find_element_by_name("password")  # Find the query box
# submit = browser.find_element_by_xpath('//*[@id="login"]')
# # submit = browser.find_element_by_xpath('//*[@id="loginForm"]/fieldset/div[2]/button')
# time.sleep(2)
# elem1.send_keys("202061222199")
# elem2.send_keys("1748young")
# submit.click()
# time.sleep(2)
#
# browser.get("https://i.njtech.edu.cn/personal/center")  # Load page
# time.sleep(5)
#
# # basicInfo = browser.find_element_by_xpath('//*[@id="block-personal"]/div/div[2]/div[2]/div[1]/div[2]')
# userName = browser.find_element_by_xpath('//*[@id="block-personal"]/div/div[2]/div[2]/div[1]/div[2]/div[1]/span')
# # print(basicInfo)
# print(userName.text())
# time.sleep(7)
# browser.close()
