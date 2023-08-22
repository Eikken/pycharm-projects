import re

import lxml
from selenium import webdriver

link_url = "https://www.baidu.com/bh/dict/ydxx_8158835209873076610?tab=%E6%A6%82%E8%BF%B0&title=%E8%82%9D%E7%99%8C&contentid=ydxx_8158835209873076610&query=%E8%82%9D%E7%99%8C&sf_ref=dict_home&from=dicta"

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(link_url)

# 获取页面源代码
html_source = driver.page_source
# 重点
html = lxml.html.fromstring(html_source)
# 获取标签下所有文本
items = html.xpath("//div[@id='y_prodsingle']//text()")
# 正则 匹配以下内容 \s+ 首空格 \s+$ 尾空格 \n 换行
pattern = re.compile("^\s+|\s+$|\n")

clause_text = ""
for item in items:
    # 将匹配到的内容用空替换，即去除匹配的内容，只留下文本
    line = re.sub(pattern, "", item)
    if len(line) > 0:
        clause_text += line + "\n"
#
#
print(clause_text)