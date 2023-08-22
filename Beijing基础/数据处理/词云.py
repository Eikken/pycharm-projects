from wordcloud import WordCloud
from lxml import etree
import requests
import re
import matplotlib.pyplot as plt


response = requests.get("https://api.bilibili.com/x/v1/dm/list.so?oid=71008195")
content = response.content.decode("utf-8")

pattern = re.compile("<d.*?>(.*?)</d>")
result = pattern.findall(content)

for i in result:
    with open('danmu.txt', 'a' , encoding='UTF-8') as f:
        f.write(i+'\n')

back_img = plt.imread(r'C:\Users\Administrator\Desktop\y_.jpg')

f = open(r'danmu.txt','r',encoding='UTF-8').read()

wc = WordCloud(
    # background_color='white',
    mask=back_img,
    font_path=r'C:\Users\Administrator\Desktop\ZhengQingKeJingYaTi-ShouBan-2.ttf', # r表示转义
    width=1000,
    height=800,
    min_font_size=8,
    max_words=150
).generate(f) # 生成

plt.imshow(wc)
plt.axis('off')
plt.savefig('ciyun')
plt.show()













