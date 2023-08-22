import requests
from lxml import etree

link_url = "http://papers.genomics.lbl.gov/cgi-bin/litSearch.cgi?query=WP_003262958.1&Search=Search"

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.26 Safari/537.36 Core/1.63.6799.400 QQBrowser/10.3.2908.400'
}

response = requests.get(url=link_url,headers=headers)
tree = etree.HTML(response.content.decode("UTF-8"))
xpath_list = tree.xpath('/html/body/ul')
for num in range(1,6):
    name = xpath_list[num].xpath('./li/a/text()')
    href = xpath_list[num].xpath('./li/a/@href')
    print("name:", name, "\nURL:", href)

