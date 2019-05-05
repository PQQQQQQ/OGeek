# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 15:58:56 2018

@author: PQ
"""

import Levenshtein
import difflib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import json

train_data = pd.read_table('oppo_round1_train_20180929.txt', 
                      names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
val_data = pd.read_table('oppo_round1_vali_20180929.txt', 
                      names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
test_data = pd.read_table('oppo_round1_test_A_20180929.txt', 
                      names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)

train_data = train_data[train_data['label'] != '音乐' ]

data = pd.concat([train_data,val_data,test_data],axis=0,ignore_index=True)

print('data shape:',data.shape)

'''
计算prefix的长度

'''
def prefix_len(x):
    try:
        return len(x)
    except:
        return len(str(x))



data['prefix_len']=data.prefix.apply(prefix_len)

'''
对query_prediction进行处理
返回长度，以及前十的概率没有就是0
返回相似度 
'''

def extract_prob(pred):
    pred = eval(pred)
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    pred_prob_lst=[]
    for i in range(10):
        if len(pred)<i+2:
            pred_prob_lst.append(0)
        else:
            pred_prob_lst.append(pred[i][1])
    return pred_prob_lst


def extract_similarity(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    len_prefix = lst[0]
    similarity=[]
    for i in range(10):
        if len(pred)<i+2:
            similarity.append(0)
        else:
            similarity.append(len_prefix/float(len(pred[i][0])))
    return similarity

def levenshtein_similarity(str1,str2):
    return Levenshtein.ratio(str1,str2)

def get_equal_rate(lst):
    pred = eval(lst[1])
    pred = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    equal_rate=[]
    for i in range(10):
        if len(pred)<i+2:
            equal_rate.append(0)
        else:
            equal_rate.append(levenshtein_similarity(lst[0],pred[i][0]))
    return equal_rate

data['pred_prob_lst']=data['query_prediction'].apply(extract_prob)
data['similarity']=data[['prefix_len','query_prediction']].apply(extract_similarity,axis=1)
data['equal_rate']=data[['title','query_prediction']].apply(get_equal_rate,axis=1)

def add_pred_similarity_feat(data):
    for i in range(10):
        data['prediction'+str(i)]=data.pred_prob_lst.apply(lambda x:float(x[i]))
        data['similarity' + str(i)] = data.similarity.apply(lambda x: float(x[i]))
        data['equal_rate' + str(i)] = data.equal_rate.apply(lambda x: float(x[i]))
    return data
data=add_pred_similarity_feat(data)

'''
对 title 进行处理
跟prefix 的相关性
跟prediction的相关性
'''
def prefix_title_sim(lst):
    return lst[0]/float(len(lst[1]))

data['prefix_title_sim']=data[['prefix_len','title']].apply(prefix_title_sim,axis=1)

# import os
# import tqdm

# def char_cleaner(char):
#     if not isinstance(char, str):
#         char = "null"

#     pattern = re.compile("[^a-zA-Z\u4E00-\u9FA5 ]")
#     char = re.sub(pattern, "", char)
#     char = char.lower()
#     return char


# def char_list_cheaner(char_list, stop_words=None):
#     new_char_list = list()
#     for char in char_list:
#         if len(char) <= 1:
#             continue
#         if stop_words and char in stop_words:
#             continue
#         new_char_list.append(char)

#     return new_char_list

# data["prefix"] = data["prefix"].apply(char_cleaner)
# data["title"] = data["title"].apply(char_cleaner)

def is_prefix_in_title(prefix, title):
    #print(prefix)

    #print(title)
    return title.find('prefix')

data['is_prefix_in_title'] = data.apply(lambda row: is_prefix_in_title(row['prefix'], row['title']), axis = 1)

data["query_prediction"] = data["query_prediction"].apply(json.loads)

data['query_prediction_len'] = data['query_prediction'].apply(lambda x:str(x).count(',') + 1)

data['title_len'] = data.title.apply(lambda x:len(str(x)))

data['title-prefix_len'] = data.title_len - data.prefix_len


# def split_num_list(pre_list):
#     # 将pred_list的概率也组合在一个列表中
#     num_list=[]
#     for string in pre_list:
# #        if string=='':
# #           num_list.append(0)
#         if len(string)!=0:
#             # 将string切成列表形式
#             #print(string)
#             s=string.replace('"','').split(': ')
#             num_list.append(float(s[-1]))
#         else:
#             num_list.append(float(0))
#     return num_list

# #split query_prediction to list
# def split_prediction(text):
#     if pd.isna(text): return []
#     return [s.strip() for s in text.replace("{", "").replace("}", "").split("\",")]

# data['pred_list']=data['query_prediction'].apply(split_prediction)
# data['pred_num']=data['pred_list'].apply(split_num_list)

# data['query_prediction_max'] = data['pred_num'].apply(lambda x: max(x))

def get_max_query_prediction(x):
    if len(x) == 0:
        return 0
    x = x.values()
    return max(x)

# for columns in data['pred_num']:
#     data['query_prediction_max'].append(columns)

data['query_prediction_max'] = data['query_prediction'].apply(lambda x:get_max_query_prediction(x))


# data['query_prediction_min'] = data['pred_num'].apply(lambda x: min(x))

# data.query_prediction_max = data.query_prediction_max.astype(float)

import re

def remove_cha(x):
     
    x = re.sub('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', "", str(x))
    x = x.replace('2C', '')
    return x

def get_query_prediction_keys(x):
    
    x = x.keys()
    x = [remove_cha(value) for value in x]
    
    return ' '.join(x)

data['query_prediction_keys'] = data.query_prediction.apply(lambda x:get_query_prediction_keys(x))

def len_title_in_query(title, query):
    query = query.split(' ')
    if len(query) == 0:
        return 0
    l = 0
    for value in query:
        #print(value)
        if value.find(title) >= 0:
            #print(l)
            l += 1
    # print(l)
    return l

data['is_title_in_query_keys'] = data.apply(lambda row:len_title_in_query(row['title'], row['query_prediction_keys']),axis = 1)

def get_query_predictions_values(x, index):
    
    length = len(x)
    if length <= index:
        return 0
    else:
        x = x.values()
        x = [float(value) for value in x]
        x = sorted(x)
        return x[index]
        
data['query_predictions_values_1'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 1))

data['query_predictions_values_2'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 2))
data['query_predictions_values_3'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 3))
data['query_predictions_values_4'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 4))
data['query_predictions_values_5'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 5))
data['query_predictions_values_6'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 6))
data['query_predictions_values_7'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 7))
data['query_predictions_values_8'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 8))
data['query_predictions_values_9'] = data.query_prediction.apply(lambda x:get_query_predictions_values(x, 9))


def get_query_predictions_keys_len(x, index):
    length = len(x)
    x = x.keys()

data['prefix_title_tag'] = data['prefix'].astype(str)+'_'+data['title'].astype(str)+'_'+data['tag'].astype(str)

a = data[data.label != -1].prefix_title_tag.value_counts().to_dict()
data['prefix_title_tag_len'] = data.prefix_title_tag.apply(lambda x:a[x] if x in a.keys() else 0)

data['sum_query_prediction_value'] = data.query_prediction.apply(lambda x:sum([float(v) for v in x.values()]))


data= data.drop(['pred_prob_lst','similarity','equal_rate'],axis=1)

data = data.drop(['prefix','query_prediction','title','label','tag'],axis=1)

print('data shape:',data.shape)

data.to_csv('other_feature.csv',index=False)