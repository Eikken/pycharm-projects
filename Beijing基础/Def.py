#def函数部分 第二天
# def add(a,b): #冒号表示标记这是函数
#     return a+b;
#
# def calculate(a,b):
#     add = a+b
#     sub = a-b
#     return add,sub #Python可返回多个值，以元组的形式
#
# #print(add(2,3))
# x,y = calculate(3,2)
# #print(x,y)
#
#
# a = 3
# b = 4

# def swap(a,b):
#     # t = a
#     # a = b
#     # b = t
#     return b,a
#
# print(a,b)
# print(swap(a,b))
# a,b = b,a #不改变a,b
#
# print(a,b)

# def change(a): #形参改变不影响实参
#     a = a + 3
#     return a
# a = 6
# print(a)
# print(change(t))

# a = [1,2]
# def cal(a): #列表传的是引用
#     a.append(5)
#     return a
# cal(a)
# print(a)
# print(cal(a))

#关键字参数
# def introduce(name,age):
#     print("我叫",name,",我今年",age,"岁了！")
#
# introduce("Eikken",19)
# introduce(age=21,name="亚索") #使用关键字赋值可以不按函数顺序赋值
#

# #默认参数
# def default_value(name,age,sex="女孩"):
#     print("我叫", name, ",我今年", age, "岁了！")
#     print("我是",sex)
#
# default_value("艾希",18,"女孩")
# default_value("卡莎",20)
# default_value("菲奥娜",24)
# default_value("奥巴马",30,"男孩")

# # 可变参数
# def change_able(*args):
#     print("可变参数：",end='')
#     for i in args:
#         print(i,end=' ')
#
# change_able()
# print()
# change_able(1)
# print()
# change_able(1,2,3) #这些都是元组
# print()
# def zi_dian(**args): #两个*传递一个字典，返回字典的toString（）
#     print(args)
#
# zi_dian()
# zi_dian(name="Ya suo",age=18)
# zi_dian(name="蕾欧娜",age=19,address="巨神峰")

#集合复习
# import time
# a = [1,2,3,4,5]
# flag = 20
# while flag:
#     t = a[0]
#     i = 0
#     while i < len(a) - 1:
#         a[i] = a[i+1]
#         i+=1
#     a[i] = t
#     print(a)
#     flag -= 1
#     time.sleep(1)
# print("结束")
# a = [[1,2,3],[4,5],[6,7,8,9]]
# i = 0
#
# while i<len(a):
#     j = 0
#     while j<len(a[i]):
#         print(a[i][j],end=' ')
#         j+=1
#     print()
#     i+=1

# a = {
#     "001":{"name":"张三","age":17,"address":"北京"},
#     "002":{"name":"李四","age":18,"address":"上海"},
#     "003":{"name":"王五","age":19,"address":"山东"},
#     "004":{"name":"Eikken","age":20,"address":"NewYork"}
# }
#
# i = 0
# b = []
# for key1,value in a.items():
#     value['num'] = key1
#     b.append(value)
#     i+=1
#
# xiao = {}
# a_ = {}
# j = 0
# while j < len(b):
#     xiao = {}
#     for k,v in b[j].items():
#
#         if k != "num":
#             xiao[k] = v
#         #print(xiao)
#         if k=="num" :
#             a_[v] = xiao
#             break
#     j+=1
# # for k,v in a_.items():
# #     print(k,v)
# print("{")
# for k in a_.keys():
#     print(" ",k,":",a_[k])
# print("}")


#print(b)
# i = 0
# for key,value in a.items():
#     print(key,":",end=' ')
#     for k,v in value.items(): #可见value == 下面的a[key1]
#         print(k,end=":")
#         print(v,end=', ')
#     print()

# for key1 in a.keys(): #第一个字典索引
#     print(key1,end=':')
#     for key2 in a[key1].keys(): #''',value in a[key1].items()''' #第二个字典索引,
#         print(key2, end=": "+str(a[key1][key2])+", ") # value = a[key]
#         # print(key2,end=": ")
#         # print(value,end=' ')
#     print()


#可变数据类型，列表、字典
# a = [1,2,3]
# def cal():
#     #a = 18 #对a无反应
#     # 函数有限读取局部变量，（无重名局部变量的情况下）能读全局变量，无法对全局变量进行赋值操作
#     # 顾名思义，对于Python来说，当你在函数内对一个重名变量进行赋值操作，等于对这个局部变量的声明
#     a.append(4) #内部修改导致a改变
# cal()
# print(a)
# a = 3 #全局变量
# def han_shu1():
#     b = 4 #嵌套变量
#     def han_shu2():
#
#         global a #声明a为全局变量
#         a = 6
#         c = 5 #局部变量
#         print(a,b,c,abs) #函数为内置
#     han_shu2()
# han_shu1()
# 寻找变量法则 l e g b
# Local 局部 Enclose 嵌套 Global 全局 Built-in 内置
# a = 10
# def han_shu():
#     a = 10
#     def nei_bu():
#         nonlocal a #外层非全局
#         a = 200
#     nei_bu()
#     print(a)
# han_shu()
#
# def di_gui(num):
#     if num != 1:
#         return num*di_gui(num-1)
#     if num == 1:
#         return num
#
# print("求阶乘")
# while True :
#     num = int(input("请输入一个数字"))
#     if num == 0: break
#     print(di_gui(num))
# a = {"name":"张三","age":20,"address":"瓦罗兰"}
# x = lambda y:y["age"]
# print(x(a))
# a = [1,1,1,1,2,3,3,3,4]
# for i in set(a):
#     a.count(i)
# m = max(a,key=a.count)  # key是以什么为关键字进行判断
# print(m)
#
# a = [-3,4,9,-1,-6]
# a.sort(key=abs)
# print(a)
#
# def han_shu(d):
#     return d['age']
# a = [
#     {"name":"张三","age":17,"address":"北京"},
#     {"name":"李四","age":28,"address":"上海"},
#     {"name":"王五","age":39,"address":"山东"},
#     {"name":"Eikken","age":20,"address":"NewYork"}
# ]
# #b = max(a,key=han_shu) #广泛用于数据分析
# b = max(a,key=lambda x:x["age"])
# print(b)

# # 函数中的变量要在函数调用前声明
# t = lambda k:k*2
# print(t) #打印函数地址

# a = [lambda x:x*i for i in range(3)]
# # i 未定义
# print(a[0](2))
# print(a[1](3))
# print(a[2](4))

# def han_shu1(b):
#     a = 3
#     def han_shu2(c):
#         print(a+b+c)
#     return han_shu2
#
# x = han_shu1(1)
# # 现在x就是han_shu2
# han_shu1(1)(2)  # 函数的闭包
# x(3)            # 闭包形成的条件
# x(4)            # 1、嵌套函数
#                 # 2、内部函数用到了外层函数的变量
#                 # 3、外层函数返回了内部函数的函数名
#                 # # # # # # # # # # # # # # # # # # #
# import time
# def cal():
#
#     sum_ = 2
#     i=0
#     while i<100000 :
#         sum_+=i
#         i+=1
#     #a = [1,2,3]
#     #print(sum(a))
#
# start = time.time()
# cal()
# end = time.time()
# print(end-start)
# import time
# def Main(func):
#     print(3)
#     def nei_bu():
#         print(4)
#         s = time.time()
#         func()
#         e = time.time()
#         print(e-s)
#         print(5)
#     return nei_bu
#
# @Main #装饰器
# def cal():
#     print(1)
#     sum_ = 0
#     i = 0;
#     while i < 3:
#         sum_+=1
#         i+=1
#     print(2)
# cal()

def b(func):
    def c ():
        print("'''''''''''''''''''''''")
        func()
        print("'''''''''''''''''''''''")
    return c

def d(func):
    def e():
        print("''''''''")
        func()
        print("''''''''")

@b #该处是返回了内层函数
@d
def a():
    print("Welcome!")

a() #运行函数是调用装饰器即调用内层函数，并把当前函数作为参数传入


