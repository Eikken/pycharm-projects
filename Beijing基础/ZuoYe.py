# 输出字符串

# while True :
#     str = input("请输入一串字符：")
#     rows = 1;
#     count = 0;
#     while len(str)-count>0:
#         for i in range(count,count+rows):#当前输出的内容为已输出的count到count+当前行数rows
#             if i == len(str) : break #数组下标越界即退出
#             print(str[i],end='')
#         print()
#         rows += 1 #行加一
#         count += rows-1 #count 为已输出的字符下标，鉴于左闭右开




#输出倒立三角形


# while True:
#     rows = int(input("输入行数："))
#     if rows == 0 :break
#     #每行输出 2*row+1 个*
#     row = 0
#     while rows > 0:
#         for i in range(0,row):
#             print(" ",end='')
#         for i in range(0,2*rows-1):
#             print("*",end='')
#
#         print()
#         row+=1
#         rows -= 1

#输出正立三角形
# while True:
#     rows = int(input("输入行数："))
#     if rows == 0 :break
#     #每行输出 2*row+1 个*
#     row = 0
#     while rows > 0:
#         for i in range(0,rows):
#             print(" ",end='')
#         for i in range(0,2*row+1):#当前行输出与行rows有什么关系
#             print("*",end='')
#
#         print()
#         row+=1
#         rows -= 1


#输出空心三角形
# while True:
#     rows = int(input("输入行数："))
#     if rows == 0 :break
#     #每行输出 2*row+1 个*
#     row = 0
#     while rows > 0:
#         for i in range(0,rows):
#             print(" ",end='')
#         for i in range(0,2*row+1):#当前行输出与行rows有什么关系
#             if rows == 1:
#                 print("*", end='')
#             elif i==0 or i==2*row :
#                 print("*",end='')
#             else:
#                 print(" ",end='')
#
#         print()
#         row+=1
#         rows -= 1


#输出倒立空心三角形
while True:
    rows = int(input("输入行数："))
    if rows == 0 :break
    #每行输出 2*row+1 个*
    row = 0
    while rows > 0:
        for i in range(0,row):
            print(" ",end='')
        for i in range(0,2*rows-1):#当前行输出与行rows有什么关系
            if row == 0:
                print("*", end='')
            elif i==0 or i==2*rows-2 :#控制行与已输出的关系
                print("*",end='')
            else:
                print(" ",end='')

        print()
        row+=1
        rows -= 1
