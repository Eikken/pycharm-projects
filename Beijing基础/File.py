# 文件类操作
# f = open("a.txt",'r')
# # f.write("hello \n")
# # f.write("world \n")
# # f.write("中国！\n")
#
# #content = f.read(2)
# content = f.readlines() # 以列表的形式读出来
# print(content)
# content = f.readline()
# print(content)
# f.close()

#
import os

path = input("请输入要建立的路径：")
cwd = os.getcwd()
subPathList = path.split("\\")
print(subPathList)
createPath = ''
for singlePath in subPathList:
    createPath = os.path.join(createPath,singlePath) #把当前目录和single join起来
    #print(createPath)
    if not os.path.exists(createPath):
        os.mkdir(createPath)
        print(createPath,"目录已创建")
    else:
        print(createPath,"目录已存在")

# while True:
#
#     order = input("请输入指令:")
#     if order=="exit":
#         break
#     elif order=="mkdir":
#         str = input("请输入目录:")
#         dire = '' #this is 动态加载目录
#         for s in str.split("\\"):
#             if os.path.exists(os.path.join(dire,s)):
#                 print(s, "目录已存在")
#                 dire = os.path.join(dire,s)
#             else:
#                 os.mkdir(os.path.join(dire,s))
#                 print(s, "目录已创建")
#                 dire = os.path.join(dire, s)
#                 print(os.path.join(dire))
#
#         print("当前根目录：",os.getcwd())
#         continue
#
#     elif order=="redir":
#         str = input("请输入目录:")
#         for s in str.split("\\"):
#             if os.path.exists(s):
#                 os.rmdir(s)
#                 print(s,"目录已删除")
#             else:
#                 print(s,"目录不存在")
#         continue







                # if os.path.exists("aa"):
#     print("已经存在该目录")
# else:
#     os.mkdir("aa",0x777)
# print(os.getcwd())