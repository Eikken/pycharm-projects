import os

def renameFile():
    fileList = os.listdir(r"D:/edu/njtech/tezhengyangpin/test") #获取文件夹下所有文件名返回list
    print(fileList)
    # get current work path
    currentpath = os.getcwd()
    print("Current is "+currentpath)
    # change current work path
    os.chdir(r"D:/edu/njtech/tezhengyangpin/test")
    i = 1
    for fileName in fileList:
        print("Original is " + fileName)
        # delete 0123456789 in file name
        lect_table = ''.maketrans('xcsdf', '12345')
        os.rename(fileName, fileName.translate(lect_table))
        print("Changed is " + fileName.translate(lect_table))
        i+=1
    os.chdir(currentpath)
renameFile()