'''
Description: 
Author: wenzhe
Date: 2023-04-26 20:06:05
LastEditTime: 2023-04-26 21:52:30
Reference: 
'''
import joblib
import jieba
import re
from sklearn.metrics import accuracy_score
text = '很好吃，以后还会买'
text = '很伤心，不会再来了'
text = '大家鼓起掌来'
text = '什么啊，好恶心'
text = '我很生气'
# text = '现在的医生真是太可怕了'
# text = '你真漂亮，能推荐一下你从哪里买的衣服吗'
text = '哇哦，你好厉害'
# text = '很高兴见到你'
# 分词工作
word_list = jieba.cut(text)
new_text = ''
# 去除符号
for word in word_list:
    word = re.sub('([^\u4e00-\u9fa5])','',word)
    # print(word)
    if len(word)!=0:
        new_text = new_text+' '+word
new_text= [new_text,]
print(new_text)
# 提取特征，计算TFIDF
tfidf_model = joblib.load('./saved_model/TFIDF_2.model')
text_feature = tfidf_model.transform(new_text)
# 载入模型
emotion_model = joblib.load('./saved_model/nb_model_2.model')
result = emotion_model.predict(text_feature)

print(result)
if result==1:
    print("Positive")
else:
    print("Negative")

