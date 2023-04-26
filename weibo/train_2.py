'''
Description: 
Author: wenzhe
Date: 2023-04-25 15:18:10
LastEditTime: 2023-04-26 21:15:28
Reference: 
'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
import joblib

# 读取数据
data = pd.read_csv('pre_2.csv')
print(len(data['labels']))
nan = float('nan')
for i in range(0,len(data['labels'])-1):
    if (data.loc[i,'words'])==nan:
        print(i)
data['words'].dropna()
print(len(data['labels']))
# data.to_csv('hello.csv')
# 划分训练集和测试集
train_data =data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)
# 测试
print("测试样本为",len(test_data['labels']))
print("积极样本为",test_data['labels'].sum())
# print(test_data)
# 提取特征
# 第一个fit_transform为边训练，边得出求解结果；
# 第二个transform为使用第一个训练得到的模型来对测试集求解
# 因此保存TFIDF模型，方便测试计算得分
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_data['words'])
joblib.dump(vectorizer,'./saved_model/TFIDF_2.model')
test_features = vectorizer.transform(test_data['words'])
# print(train_features)

# # 使用朴素贝叶斯分类器 0.80 0.80 0.80 0.80
# # 开始训练
# clf_nb=MultinomialNB()
# clf_nb.fit(train_features,train_data['labels'])
# # 测试集测试
# pred = clf_nb.predict(test_features)
# joblib.dump(clf_nb,'./saved_model/nb_model_2.model')

# # 使用svm分类器   0.86 0.86 0.86 0.86
# clf_svm = LinearSVC()
# clf_svm.fit(train_features,train_data['labels'])
# # 测试集测试
# pred = clf_svm.predict(test_features)
# joblib.dump(clf_svm,'./saved_model/svm_model_2.model')

# # 使用 knn算法  0.56 0.68 0.56 0.47
# clf_knn = KNeighborsClassifier()
# clf_knn.fit(train_features,train_data['labels'])
# pred = clf_knn.predict(test_features)  
# joblib.dump(clf_knn,'./saved_model/knn_model_2.model')

# 使用random forest算法：复杂度太高，半天才算出来  0.996
clf_rf = RandomForestClassifier()
clf_rf.fit(test_features,test_data['labels'])
print("train end")
pred = clf_rf.predict(test_features)
# 保存模型
joblib.dump(clf_rf,'./saved_model_2/rf_model.model')

# 计算评估参数
acc = accuracy_score(test_data['labels'],pred)
pre = precision_score(test_data['labels'],pred,average='weighted')
recall= recall_score(test_data['labels'],pred,average='weighted')
f1 = f1_score(test_data['labels'],pred,average='weighted')

print(acc)
print(pre)
print(recall)
print(f1)
