'''
Description: 
Author: wenzhe
Date: 2023-04-26 19:55:17
LastEditTime: 2023-04-26 23:40:10
Reference: 
'''
import pandas as pd

# 读取有内容的行
# fp = open('text.txt','r',encoding='utf8')
# lines = fp.readlines()
# with open('./text2.txt','w+',encoding='utf-8') as file:
#     for line in lines:
#         if line[1]=='s':
#             file.write(line)
# 第二部分把内容行，分解为标签和语句
fp = open('text2.txt','r',encoding='utf8')
lines= fp.readlines()
labels=[]
types =[]
sentences=[]
for line in lines:
    # 中性还是其他
    point = line.find('emotion_tag')
    label = line[point+13]
    labels.append(label)
    # 如果不是中性，具体是什么
    if label=='N':
        types.append('nothing')
    else:
        point = line.find('emotion-1-type')
        print(line[point+16]+line[point+17]+line[point+18])
        types.append(line[point+16]+line[point+17]+line[point+18])
    # 将内容存储进来
    left = (line.split('>'))
    right = left[1].split('<')
    sentences.append(right[0])

labels=pd.DataFrame(labels)
sentences=pd.DataFrame(sentences)
# 0 为中性，1 为positive，-1为negative。
labels_num=[]
print(labels)
print(labels[0][1])
for i in range(0,len(labels)):
    if labels[0][i] == 'N':
        labels_num.append(0)
    else:
        if types[i]=='lik' or types[i]=='hap' or types[i]=='sur':
            labels_num.append(1)
        elif types[i]=='sad' or types[i]=='ang' or types[i]=='dis' or types[i]=='fea' or types[i]=='" e':
            labels_num.append(-1)
        else:
            print('救命啊家人们') 
            print(i)
            print(types[i])
labels_num=pd.DataFrame(labels_num)
print(labels_num)
print(sentences)
# print(labels)