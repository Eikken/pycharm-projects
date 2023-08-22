import requests
from lxml import etree

link_url = "https://hr.tencent.com/position.php?&start=%d#a"
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
}

for page in range(1,3):
    new_link = link_url%((page-1)*10)

    response = requests.get(url=new_link,headers=headers)
    tree = etree.HTML(response.content.decode("UTF-8"))
    # //*[@id="position"]/div[1]/table 的Xpath
    # //*[@id="position"]/div[1]/table/tbody/tr[2]/td[1]/a
    # //*[@id="position"]/div[1]/table/tbody/tr[2]
    # //*[@id="position"]/div[1]/table/tbody/tr[2]/td[3]
    # //*[@id="position"]/div[1]/table/tbody/tr[2]/td[2]
    xpath_list = tree.xpath('//table[@class="tablelist"]/tr')
    # print(xpath_list)
    flag = 1
    for xp in xpath_list:
        if flag == 1:
            flag += 1
            continue
        if flag == 12:
            flag += 1
            continue
        name = xp.xpath("./td/a/text()")
        type = xp.xpath("./td[2]/text()")
        num = xp.xpath("./td[3]/text()")
        address = xp.xpath("./td[4]/text()")
        date = xp.xpath("./td[5]/text()")
        #print(name,type,num,address,date)
        flag += 1
        info = name[0]+"  "+type[0]+"  "+num[0]+"  "+address[0]+"  "+date[0]
        with open("Tencent.txt",'a',encoding="UTF-8") as f:
            f.write(info+'\n')
            f.close()
    print("写完了")