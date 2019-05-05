# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:29:53 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import string
import jieba


df_train = pd.read_table('oppo_round1_train_20180929.txt', 
                      names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
df_valid = pd.read_table('oppo_round1_vali_20180929.txt',
                    names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)
df_testa = pd.read_table('oppo_round1_test_A_20180929.txt',
                    names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)


df_train = df_train[df_train['label']  != '音乐']

df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)

df_testa['label']=-1
        
df_data=pd.concat([df_train,df_valid,df_testa]).reset_index(drop=True)

str_mapper = str.maketrans("","",string.punctuation)

df_data['cut_prefix'] = df_data['prefix'].apply(lambda x:jieba.lcut(x.translate(str_mapper)))
df_data['cut_title'] = df_data['title'].apply(lambda x:jieba.lcut(x.translate(str_mapper)))
df_data['cut_query_prediction'] = df_data['query_prediction'].apply(lambda x:{k:[[float(v)],jieba.lcut(k.translate(str_mapper))] for k,v in eval(x).items()})

def func(x):
    text = []
    text.extend(x['cut_prefix'])
    text.extend(x['cut_title'])
    preds=  []
    for k,v in x['cut_query_prediction'].items():
        preds.extend(v[1] )
    text.extend(preds)
    return text

df_data['words']=df_data.apply(lambda x:func(x),axis=1)


def func1(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            score +=v[0][0]
    return score

df_data['title_pred_score']=df_data.apply(lambda x:func1(x),axis=1)


def func2(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    nwords = 0.0
    new_words= []
    pred_words = []
    for k,v in query_prediction.items():
        pred_words.extend(v[1])
    new_words=[word for word in title_words if word not in pred_words]
    return len(new_words)

df_data['title_unseen_nword']=df_data.apply(lambda x:func2(x),axis=1)


def func3(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    std = np.std(freqs) if len(freqs)>0 else 0.0
    return std

df_data['pred_freq_std']=df_data.apply(lambda x:func3(x),axis=1)


def func4(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    std = np.mean(freqs) if len(freqs)>0 else 0.0
    return std

df_data['pred_freq_mean']=df_data.apply(lambda x:func4(x),axis=1)


def func5(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    return len(title_words)

df_data['nword_title_unseen_in_prefix']=df_data.apply(lambda x:func5(x),axis=1)


def func6(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    _sum = np.sum(freqs) if len(freqs)>0 else 0.0
    return _sum

df_data['pred_freq_sum']=df_data.apply(lambda x:func6(x),axis=1)


def func7(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    fit_nwords=[]
    fit_keys = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            fit_nwords.append(len(pred_words))
            fit_keys.append(k)
    if len(fit_keys)>0:
        k = fit_keys[np.argmax(fit_nwords)]
        score = query_prediction[k][0][0]
    return score

df_data['title_unseen_in_prefix_score_max']=df_data.apply(lambda x:func7(x),axis=1)


def func8(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    
    scores = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            scores.append(v[0])
    std = np.std(scores) if len(scores)>0 else 0.0
    return std

df_data['title_unseen_in_prefix_score_std']=df_data.apply(lambda x:func8(x),axis=1)


def func9(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    
    scores = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            scores.append(v[0])
    mean = np.mean(scores) if len(scores)>0 else 0.0
    return mean

df_data['title_unseen_in_prefix_score_mean']=df_data.apply(lambda x:func9(x),axis=1)

df_data['prefix_nwords']=df_data['cut_prefix'].apply(lambda x: len(x))


df_data.to_csv('text_df.csv', index=False)
