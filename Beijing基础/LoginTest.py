import requests
from lxml import etree
headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36"}
url="http://www.renren.com/PLogin.do"
data = {
    'email': '17863946029',  #
    'password': '**********'  #
}
x=requests.post(url,data=data,headers=headers)
print(x.content.decode('utf-8'))