'''
Description: 
Author: wenzhe
Date: 2023-04-26 15:35:39
LastEditTime: 2023-04-26 19:42:53
Reference: 
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:28:41 2020

@author: cm
"""

from networks import SentimentAnalysis
import pandas as pd

SA = SentimentAnalysis()


def predict(sent):
    """
    1: positif
    0: neutral
    -1: negatif
    """
    score1,score0 = SA.normalization_score(sent)
    if score1 == score0:
        result = 0
    elif score1 > score0:
        result = 1
    elif score1 < score0:
        result = -1
    return result
        

if __name__ =='__main__':
    # 直接使用情绪词典，效果不好
    data = pd.read_csv('weibo_senti_100k.csv')
    labels = []
    count = 0
    for i in range(0,len(data['label'])-1):
        labels.append(predict(data.loc[i,'review']))
    for i in range(0,len(data['label'])-1):
        if labels[i] ==  data.loc[i,'label']:
            count=count+1
    
    print(count/len(data['label']))



