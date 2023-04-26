'''
Description: 
Author: wenzhe
Date: 2023-04-26 11:20:25
LastEditTime: 2023-04-26 21:35:44
Reference: 
'''
import pandas as pd
import numpy as np
import jieba
import re 

data = pd.read_csv('weibo_senti_100k.csv')
# print(data['label'])
print(len(data['review']))
# 创建新容器，存储分词结果
part = pd.DataFrame(data=data['label'])
part.insert(loc=1,column='words',value='nothing')
stopwords = [line.strip() for line in open('hgd_stopwords.txt','r',encoding='utf8').readlines()]

for i in range(1,len(data['review'])):
    # 使用jieba分词
    word_list = jieba.cut(data.loc[i-1,'review'])
    new_text = ''
    # 去除符号
    for word in word_list:
        word = re.sub('([^\u4e00-\u9fa5])','',word)
        # print(word)
        if len(word)!=0:
            if word not in stopwords:
                new_text = new_text+' '+word
    if new_text == '':
        new_text='NaN'
    # print(new_text)
    part.loc[i-1,'words']=new_text
part.to_csv('pre_3.csv')
print(part)
