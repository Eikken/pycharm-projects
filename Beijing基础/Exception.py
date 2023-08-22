# try:
#     a = [1,2,3]
#     i = 0
#     print(3/0)
#     while i < 5:
#         print("a[%d]"%i,a[i])
#         i+=1
#
# except :
#     print("各种异常")
# else:
#     print("没有异常")
# finally:
#     print("finally 语句")

# except IndexError as e:
#     print("异常：",e)
# except ZeroDivisionError as e:
#     print("异常2",e)
# else:
#     print("没有异常")
#抛出异常 raise

import re
#str_ = ",_901abc"
#result = re.compile("\W").match(str_)
#print(result.group())
# \d 匹配一个数字 ；\D 匹配除数字外的字符
# \w 匹配一个英文字母或一个数字或一个下划线 re.A 是匹配Asc码
# \W 匹配出数字字符下划线以外的 \s 匹配一个空白字符如（\n \r \f \t）\tab

# 原子表
# [字母] 仅字母中所包含的 ；[^字母] 所有的除了字母中所包含的
# * 匹配0次、一次或多次*前面的原子
# + 匹配一次或多次其前面的字符
# ? 匹配0次或一次?前面的字符
# | 匹配或关系，满足或即可在字符串中匹配到
# ^与$ 在不用在中括号中时表示字符串的开头和结尾的限定标志
# {m} 其前面的原子出现m次,{m,n}表示其前面的原子至少出现m次，最多n次 ，{m,}表示其前面的原子至少出现m次
# () 整体表示一个原子，只要这个原子中的内容
# r"\\\\" 表示正则表达式内使用原始的字符不让转义
# . 除了换行符全都匹配
# str_ = "prey901abc"
# result = re.compile("^]").match(str_)
# print(result)
# c = "122223bc"
# r = re.compile("^\d*").findall(c)
# r = re.compile("12{2}").findall(c)
# r = re.compile("12{2}").findall(c)
# c = "电话：010-12345 电话：053912345678"
# r = re.compile("：(\d{3})-(\d*)").findall(c)
# s = re.compile("：(\d{4})(\d*)").findall(c)
# print(r,s)

# c = "abbbbc"
# result = re.compile("ab?").match(c)
# print(result)
#
# c = "\\\\"
# result = re.compile(r".").match(c).group()#r"\\\\").match(c).group()
# print(result)
# .*? 表示非贪婪，.*表示贪婪通配，一配到低
html = "<div>我爱你北京</div> <br>南京</br>"

# p = re.compile(r"<\w+>(.*?)</\w+>")
# p = re.compile(r"<(\w+)>(.*?)</\1>")  # \1 表示第一个括号内的内容又要使用 \w+尽可能多的匹配
# r = p.findall(html)  #返回一个列表
# for i in r:
#     print(i[1])
#
ip = "192.168.119.254 255.10.10.7 10.10.192.1"
pattern = re.compile(r"((\d{1,3}\.\d{1,3}\.\d{1,3}))").findall(ip)

for i in pattern:
    print(i[0])



