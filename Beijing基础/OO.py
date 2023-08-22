# class D():
#     __slots__ =  ("name","age","sex")
#     def __init__(self,name,age,sex):
#         self.name = name
#         self.age = age
#         self.sex = sex
#
#     def introduce(self):
#         print("Hello, I'm",self.name)
#         print("I'm", self.age, "years old!")
#         #print("My ID is",self.ID)
#         print("My sex is",self.sex)
#
# a = D("嘉文",23,"男")
# a.introduce()
# a.ID = 1001
# print(a.ID)


# class A():
#     count = 0
#     def __init__(self):
#         self.num = 100
#         A.count += 1
#         #pass
#     def __set_num__(self, num):
#         self.num = num
#
#     def say(self):
#         print(self.num)
#     @classmethod
#     def cMethod(cls):
#         print(cls.count)
#     @staticmethod
#     def sMethod():
#         print("这是静态方法")
#
# # 公有类变量全都可用，对象一旦被赋值，优先使用对象本身的值，对象属性与公有属性将没有任何关系了。
# a = A()
# b = A()
# A.cMethod()
# b.cMethod()
# A.say(a)
# A.sMethod()
# b.sMethod()

# a.num = 120
# delattr(a,"num") #删除属性
# print(a.num)
# a.num = 90
# print(a.num)
# b = A()
# print(b.num)
# A.num = 1000
# print(a.num)
# print(b.num)
# delattr(a,"num")
# print("a",a.num)
#class People():
#     def __set_name__(self, name):
#         self.name = name
#     def introduce(self) :
#         print("Hello, I'm",self.name)
#         print("I'm", peo.age, "years old!")
#         print("My ID is",self.ID)
#         print("My sex is",self.sex)
#     def __init__(self):
#         self.name = "Zed"
#         self.age = 18
#         self.ID = 1000
#     def Sex(self):
#         self.sex = "男"
#     def __init__(self,n,a,id): #构造参数
#         self.name = n
#         self.age = a
#         self.ID = id
#     @staticmethod
#     def static():
#         print("这是静态方法。")
# peo = People("赵四",35,1002)
# peo.Sex()
# #print(peo.sex)
# peo.introduce()
#
# peo1 = People("刘能",45,2001)
# peo1.Sex()
# peo1.introduce()

# peo.ID = 1001
# peo.name = "Yasuo"
# peo.age = 21
# peo.__set_name__("Bob")
# peo.introduce()
# peo.age = 17
# People.static()
# peo.static() #类和对象都能直接调用静态方法。
#Str = "I'm "+str(peo.age)+" years old!"#os.path.join()
#peo.introduceage = print("I'm",peo.age,"years old!")
# class Dog():
#     def jiao(self):
#         print("wang!wang!")
# class Cat():
#     def jiao(self):
#         print("miao~miao~")
# class Pig():
#     def jiao(self):
#         print("heng heng!")
#
# def jiao(obj):
#     obj.jiao()
#
# d = Dog()
# c = Cat()
# p = Pig()
#
# jiao(d)
# jiao(c)
# jiao(p)
# class Girl():
#     def __init__(self,name,age,heigh):
#         self.name = name
#         self.__age = age
#         self.heigh = heigh
#     def introduce(self):
#         print("我叫",self.name)
#         print("年龄", self.__age)
#         print("身高", self.heigh)
#
# red = Girl("xiao hong",20,150)
# red.introduce()
# print(red.__age)
# #python 双下划线__代表内部使用
#
#继承
# class A():
#     # def __init__(self,name,age):
#     #     self.name = name
#     #     self.age = age
#     pass
# class B(A):
#
#     # def __init__(self,name,age):
#     #     super().__init__(name)
#     #     self.age = age
#
#     def say(self):
#         print("I am %s"%self.name)
#         print("I am",self.age,"years old!")
#
# b = B("张三",19)
# b.say()
# class St():
#     def __new__(cls, *args, **kwargs):
#         print("First")
#         return object.__new__(cls)
#     def __init__(self,n):
#         print("Second")
#         self.name = n
# a = St("Hello")

#import Fu_Lei
# from Fu_Lei import *#add,multi
# a = int(input("a:"))
# b = int(input("b:"))
# y = add(a, b)
# print("a+b=", y)
# y = multi(a, b)
# print("a*b=", y)
# 2019-01-16 15:19:29
#
# import time
# import datetime
# q = "2019-01-15 15:19:29"
# second = time.mktime(time.strptime(q,"%Y-%m-%d %H:%M:%S")) #当天已过
#
# print(second)
#
# date = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(second))
#
# print("账户锁定日期：",date)
#
# pause = 7*24*3600
#
# print("距离解冻时间还剩：")
# flag = 0
# while True:
#     flag += 1
#     pass_time = time.time() - second
#     left_time = int(pause - pass_time)
#     day = left_time//(24*3600)
#     hour = (left_time - day*(24*3600))//3600
#     minute = (left_time - day*(24*3600) - hour*3600)//60
#     seconds = left_time%60
#
#     print("还有%d天%d时%d分%d秒解冻"%(day,hour,minute,seconds))
#     #print(time.strftime("%H:%M:%S",time.localtime(left_time)))
#     time.sleep(1)
#     if flag > 20:
#         break


#now = time.time()
#print(2019-now//3600//24//365)

# class A():
#     count = 0
#     def __init__(self):
#         A.count += 1
#     def __del__(self):
#         A.count -= 1
#         print("now run on here!",A.count,"left")
# a = A()
# b = A()
# print(A.count)
#del a
#del b
#
# class Student():
#     def __init__(self,name,age,address):
#         self.name = name
#         self.age = age
#         self.address = address
#
#     def __str__(self):
#         str  = "我叫%s 年龄%d 地址%s"%(self.name,self.age,self.address)
#         return str
#
# a = Student("张三",19,"New York")
# print(str(a))
# print(a)
# id =
# a = [34,567]
# b = [34,567]
# print(a==b)
# print(id(a))
# print(id(b))

# copy
# import copy
#
# a = [1,2,[3,4]]
# d = a
# b = copy.copy(a)
# c = copy.deepcopy(a)
# a.append(5)
# print(a)
# print(b)
# print(c)
# print(a==b)
# print(a==c)
# print(b==c)
# print(id(a[2]))
# print(id(b[2]))
# print(id(c))
# print(d)

# def Decor(func):
#     def _decor(typename):
#         print("求神拜佛，祝愿一切顺利")
#         result = func(typename) #相当于调用Date
#         print("约会成功，烧香还愿")
#         return result
#     return _decor
#
# @Decor #Date = Decor(Decor)
# def Date(typename):
#     print("带{}，吃饭、逛街、看电影".format(typename))
#     return typename+"一枚"
# #调用函数
# result = Date("萌妹子")
# print(result)