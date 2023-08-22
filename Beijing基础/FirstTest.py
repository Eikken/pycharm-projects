"""
k = a.keys
则k为a中的键值
a.get(value,default_value),获得一个value值，然后两值相加，默认无key值为0
k.lower()获得所有的键值并置为小写
print k.lower()键值的所有的加和值
"""

print()
# a="abcedfg12345678"
# print(a[2:5])#字符串切片前闭后开区间输出
# print(a[2:15:3])#字符串切片前闭后开区间输出,第三个字符为步长
# print(a.find("f"))
# #a[0]="1"
# print(a)
# #字符串是不可更改的类型
# i=0
# while i < len(a) :
#     print("a[",i,"] =",a[i])
#     i+=1
# .strip() 去掉字符串两头的空格，中间的不会去,lstrip()与rstrip()为左右去空格
# a=[8,6,3,2,1,5]
# print(type(a))
# a.insert(0,10)
# print(a)
# a[3] = 100
# print(a)
# del a[0]
# print(a)
# a.reverse()
# print(a)
# a.sort()
# print(a)
# a.sort(reverse=True)
# print(a)
#
# b=(1)
# print(type(b))
# # print(b[0])
a={"name":"张三","age":18,"address":"背景","age":20}#键值不可重复
print(a["name"])
a["sex"]="男"

c=a.get("name",'''缺省值''')
c={v:k for k,v in a.items()}
print(c)
# print(a)
# a["age"]=20
# print(a)
# print(a.keys())
# print(a.values())
# print(a.items())
# print(a.popitem())
# for key,value in a.items():
#     print(key,value,end=" ")
#
# a=set((1,2,3,3,3))
# print(a)
# b=[1,2,3,3,3]
# c=set(b)
# print(c)
# a=[i for i in range(5)]
a=[[1,2,3],[4,5,6],[7,8,9]]
b = [a[i][1] for i in range(len(a))]


#print([a[i][len(a)-i-1] for i in range(len(a))])
#以上叫推导式
#print("Hello"+" Test "+"World")
"""这是一条
多行
注释"""
#这是一条单行注释
'''这又是一条

多行
注释'''
import keyword
# print(keyword.kwlist) #这是Python的一些关键字
'''
a="3//-2"
b=3//-2
c=3/2
print("type of 'a' is ",type(a))
# // 取余符号并且向下取整 为
print(type(b))
print(type(c))
'''
 #input 输入
"""

print(sc, " 的类型是：",type(sc))
print("转换后为：",type(int(sc)))
"""
#flag = 5
'''

while flag > 1 :
    sc =int(input("请输入"))
    flag -= 1
    if sc>=90 and sc<=100:
      print("get Score:A")
    elif sc>=80:
       print("get Score:B")
    elif sc >= 70:
       print("get Score:C")
    elif sc>=60:
       print("get Score:D")
    else:print("不及格！")

a=1234
b=0
while a>0:
    b = b*10 + a%10
    a = a//10
print(b)



a = int(input("输入数字："))
i = 2
while i<a :
    if a%i == 0:
        print(a,"是非质数")
        break
        
    i+=1
else:
    print(a,"是质数")
    
for i in range(1, 10, 6):
    print(i)   
    
    
     
'''

