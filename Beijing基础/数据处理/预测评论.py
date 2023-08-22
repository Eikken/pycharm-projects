import pandas as pd
import jieba
data = pd.read_csv(r'C:\Users\Administrator\Desktop\data.csv', encoding='gbk')
# print(data.columns)
arrs = data['内容 ']
# print(col)
with open(r'C:\Users\Administrator\Desktop\stopwords.txt', 'r', encoding='UTF-8') as fp:
    stopwords = fp.readlines()
# print(stopwords)

stopwords_list = list(map(lambda word:word.strip(), stopwords))
stopwords_list = set(stopwords_list)

# 评论列表
comment_list = []
for i in arrs:
    setList = jieba.cut(i, cut_all=False)
    final = ''
    for seg in setList:
        if seg not in stopwords_list:
            final += seg
    # print('final:', final)
    # jieba 没用停用词的分词列表
    segList = jieba.cut(final, cut_all=False)
    output = ''.join(list(segList))
    comment_list.append(output)
# print(comment_list)
# 计算词频
from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer()
X = vector.fit_transform(comment_list)
word = vector.get_feature_names() # 切完的词
# print(word)
# print(X)  # X 是词语出现的次数
# print(X.toarray())
# 机器学习部分
x_train = X.toarray()[:10]  # 训练集
x_test = X.toarray()[10:]  # 测试集
arrs_comment = data['评价']  # 标签
# print(arrs_comment)

y = list(map(lambda a: 1 if a == '好评' else 0, arrs_comment))  # y就是0101集合
y_train = y[:10]
y_test = y[10:]

from sklearn.naive_bayes import MultinomialNB
# 调用朴素贝叶斯的算法
clf = MultinomialNB()
clf.fit(x_train, y_train)
result = clf.predict(x_test)
print('预测值：', result)
print('真实值：', y_test)




