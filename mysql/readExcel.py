import pandas as pd
import numpy as np
import MySQLdb
def getDataSet():
    DataSet = pd.read_excel(r'人口普查信息.xlsx')
    dataSet = np.array(DataSet).tolist()
    # columns = np.array(DataSet.columns).tolist()
    columns = ['序号', '学号', '公民身份号码', '姓名', '学院', '联系电话', '户籍地址', '现居地', '学历', '民族']
    return DataSet, dataSet, columns

def insertSql(dataSet ,col):
    db = MySQLdb.Connect("localhost", "root", "123456", "iamstudents", charset='utf8')
    cursor = db.cursor()
    dataSet[col]
    tuser = (2, '20206122211', '371325199812', '李四', 'IAM', '17863946028', '江苏省南京市鼓楼区', '新模三号楼777', '硕士', '满族')
    sql = "INSERT INTO studentinfo(seriesNumber,schoolId,identityCard,studentName,college,phoneNumber,householdRegister," \
          "currentDwelling,degree,nation) VALUES "
    try:
        cursor.execute(sql+str(tuser))
        db.commit()
        print("sql insert success !")
    except Exception as e:
        print(sql+str(tuser))
        print("sql insert fail !", e)
        db.rollback()
    db.close()
if __name__ == '__main__':
    data, list, col = getDataSet()
    subset = data[col]

    for i in subset.iloc[1]:
        pass
    
    # tuples = [tuple(x) for x in subset.values]
    # print(tuples)
    # insertSql()