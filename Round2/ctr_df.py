# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:13:53 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import json
import jieba
import utils
import time
import re
from gensim.models.keyedvectors import KeyedVectors
import warnings


warnings.filterwarnings('ignore')
t0 = time.time()

import unicodedata
from itertools import groupby
import re

def text_clean(text,patterns=[]):
    patterns = [('','')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    """ Simple text clean up process"""
    clean = text.lower()
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return clean

def _get_ctr_df(df, train_df_length):
    df = df.copy()
        
    train_df = df[:train_df_length]
    train_df['label'] = train_df['label'].apply(lambda x : int(x))
        
    labels_columns = ["prefix", "title", "tag"]
    for column in labels_columns:
        click_column = "{column}_click".format(column=column)
        count_column = "{column}_count".format(column=column)
        ctr_column = "{column}_ctr".format(column=column)

        agg_dict = {click_column: "sum", count_column: "count"}
        ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
        ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

        df = pd.merge(df, ctr_df, how="left", on=column)

    # combine features
    for idx, column1 in enumerate(labels_columns):
        for column2 in labels_columns[idx + 1:]:
            group_column = [column1, column2]
            click_column = "{column}_click".format(column="_".join(group_column))
            count_column = "{column}_count".format(column="_".join(group_column))
            ctr_column = "{column}_ctr".format(column="_".join(group_column))

            agg_dict = {click_column: "sum", count_column: "count"}
            ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
            ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

            df = pd.merge(df, ctr_df, how="left", on=group_column)
            
    # kmeans features
    labels_columns_1 = ["cate_prefix", "cate_title"]
    for column in labels_columns_1:
        click_column = "{column}_click".format(column=column)
        count_column = "{column}_count".format(column=column)
        ctr_column = "{column}_ctr".format(column=column)

        agg_dict = {click_column: "sum", count_column: "count"}
        ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
        ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

        df = pd.merge(df, ctr_df, how="left", on=column)
    
    labels_columns_2 = ["cate_prefix", "cate_title", "tag"]
    for idx, column1 in enumerate(labels_columns_2):
        for column2 in labels_columns_2[idx + 1:]:
            group_column = [column1, column2]
            click_column = "{column}_click".format(column="_".join(group_column))
            count_column = "{column}_count".format(column="_".join(group_column))
            ctr_column = "{column}_ctr".format(column="_".join(group_column))

            agg_dict = {click_column: "sum", count_column: "count"}
            ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
            ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

            df = pd.merge(df, ctr_df, how="left", on=group_column)
            
        
    group_column = ['prefix', 'title', 'tag']
    click_column_1 = "{column}_click".format(column="_".join(group_column))
    count_column_1 = "{column}_count".format(column="_".join(group_column))
    ctr_column_1 = "{column}_ctr".format(column="_".join(group_column))
    agg_dict_1 = {click_column_1: "sum", count_column_1: "count"}

    temp1 = train_df.groupby(['prefix','title','tag'], as_index=False)['label'].agg(agg_dict_1)
    temp1[ctr_column_1] = temp1[click_column_1] / (temp1[count_column_1] + 5)  # 平滑
    df = pd.merge(df, temp1, how="left", on=group_column)


    group_column_1 = ['prefix_num', 'title_num', 'tag']
    click_column_2 = "{column}_click".format(column="_".join(group_column_1))
    count_column_2 = "{column}_count".format(column="_".join(group_column_1))
    ctr_column_2 = "{column}_ctr".format(column="_".join(group_column_1))
    agg_dict_2 = {click_column_2: "sum", count_column_2: "count"}

    temp2 = train_df.groupby(['prefix_num','title_num','tag'], as_index=False)['label'].agg(agg_dict_2)
    temp2[ctr_column_2] = temp2[click_column_2] / (temp2[count_column_2] + 5)  # 平滑
    df = pd.merge(df, temp2, how="left", on=group_column_1)
    

    return df


train_df = utils.read_txt('../../data_train.txt')
validate_df = utils.read_txt('../../data_vali.txt')
test_df = utils.read_txt('../../data_test.txt',is_label=False)


train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]
df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True)

kmeans_df_1 = pd.read_csv('kmeans_prefix_x_1.csv')
kmeans_df_2 = pd.read_csv('kmeans_title_x_1.csv')

#kmeans_df = pd.read_csv('kmeans_df_x.csv')
df = pd.concat([df, kmeans_df_1,kmeans_df_2], axis=1)

df["prefix"] = df["prefix"].apply(text_clean)
df["title"] = df["title"].apply(text_clean)

df['prefix_num'] = df['prefix'].apply(lambda x:len(x))
df['title_num'] = df['title'].apply(lambda x:len(x))


# df["query_prediction"] = df["query_prediction"].apply(json.loads)

# clearn prefix and title


#填充query_prediction的缺失值
def func(x):
    preds = x['query_prediction'].unique()
    preds = [_x for _x in preds if len(_x)>0]
    if len(preds)==0:
        return ''
    else:
        return preds[0]
    
df_tmp = df.groupby(['prefix']).apply(lambda x:func(x)).reset_index(drop=False).rename(columns={0:'cleaned_query_prediction'})
df = pd.merge(df,df_tmp,on='prefix',how='left')

df = df.drop(columns=['query_prediction'])
df = df.rename(columns={'cleaned_query_prediction':'query_prediction'})

ctr_df = _get_ctr_df(df, train_df_length)

ctr_df.to_csv('ctr_df_x_1.csv', index=False)

print(time.time() - t0)