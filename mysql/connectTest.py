import pymysql

def connectDB():
    conn = pymysql.connect(host='localhost',
                           port=3306,
                           database='iamstudents',
                           user='root',
                           passwd='yxw@Sql1223',
                           charset='utf8')

    cursor = conn.cursor()

    # 使用 execute()  方法执行 SQL 查询
    cursor.execute("""SELECT
                        studentinfo.seriesNumber,
                        studentinfo.schoolId,
                        studentinfo.identityCard,
                        studentinfo.studentName,
                        studentinfo.profession,
                        studentinfo.college,
                        studentinfo.phoneNumber,
                        studentinfo.householdRegister,
                        studentinfo.currentDwelling,
                        studentinfo.degree,
                        studentinfo.nation
                        FROM
                        studentinfo""")


    # 使用 fetchone() 方法获取单条数据;使用 fetchall() 方法获取所有数据
    data = cursor.fetchall()
    for item in data:
        print(item)

    # 关闭数据库连接
    cursor.close()


if __name__ == '__main__':
    connectDB()

