import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import utils
import zipfile
from gensim.models.keyedvectors import KeyedVectors

import time
import json
import re
import jieba
import Levenshtein
import logging
import warnings
import pickle

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn import metrics


prefix_df = pd.read_csv('prefix_df_pq.csv')
ctr_df = pd.read_csv('cvr_cv_df_3.csv')
ctr_df_2 = pd.read_csv('ctr_df_pq_1.csv')
query_df = pd.read_csv('query_df_pq.csv')
text_df = pd.read_csv('text_df_pq.csv')
stat_df = pd.read_csv('stat_df_pq.csv')
#nunique_df = pd.read_csv('nunique_df_pq.csv')
com_prefix_df = pd.read_csv('comp_prefix_df.csv')
# query_df_leven = pd.read_csv('query_df_leven.csv')
#add_df = pd.read_csv('add_df_pq.csv')
#query_mutual_df = pd.read_csv('query_mutual_df.csv')
# kmeans_df_1 = pd.read_csv('kmeans_prefix_w2v_2.csv')
# kmeans_df_2 = pd.read_csv('kmeans_title_w2v_2.csv')


# drop_columns_k1 = ['label', 'prefix', 'query_prediction', 'tag', 'title', 'cutprefix', 'cutprefix_vec']
# drop_columns_k2 = ['label', 'prefix', 'query_prediction', 'tag', 'title', 'cuttitle','cuttitle_vec']
# kmeans_df_1 = kmeans_df_1.drop(columns=drop_columns_k1)
# kmeans_df_2 = kmeans_df_2.drop(columns=drop_columns_k2)
drop_columns = ['complete_prefix']
ctr_df = ctr_df.drop(columns=drop_columns)

drop_columns = ['prefix', 'title']
prefix_df = prefix_df.drop(columns=drop_columns)

drop_columns_p = ['complete_prefix', 'title']
com_prefix_df = com_prefix_df.drop(columns=drop_columns_p)

drop_columns_c = ['label', 'prefix', 'query_prediction', 'tag', 'title',
       'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
       'prefix_click', 'prefix_count', 'prefix_ctr', 'title_click',
       'title_count', 'title_ctr', 'tag_click', 'tag_count', 'tag_ctr',
       'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr',
       'prefix_tag_click', 'prefix_tag_count', 'prefix_tag_ctr',
       'title_tag_click', 'title_tag_count', 'title_tag_ctr','prefix_title_tag_click',
       'prefix_title_tag_count', 'prefix_title_tag_ctr']
ctr_df_2 = ctr_df_2.drop(columns=drop_columns_c)

# drop_columns_c = ['cate_prefix', 'cate_title','prefix_click', 'prefix_count',         
#                   'title_click','title_count',  
#        'prefix_title_click', 'prefix_title_count', 
#        'prefix_tag_click', 'prefix_tag_count', 
#        'title_tag_click', 'title_tag_count', 'prefix_title_tag_click',
#        'prefix_title_tag_count']
# ctr_df = ctr_df.drop(columns=drop_columns_c)

# drop_columns_c = ['cate_prefix', 'cate_title','prefix_click', 'prefix_count', 'prefix_ctr',         
#                   'title_click','title_count', 'title_ctr', 'tag_click', 'tag_count', 'tag_ctr',
#        'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr',
#        'prefix_tag_click', 'prefix_tag_count', 'prefix_tag_ctr',
#        'title_tag_click', 'title_tag_count', 'title_tag_ctr','prefix_title_tag_click',
#        'prefix_title_tag_count', 'prefix_title_tag_ctr']
# ctr_df = ctr_df.drop(columns=drop_columns_c)

# drop_columns = ['cate_prefix', 'cate_title']
# ctr_df = ctr_df.drop(columns=drop_columns_c)

drop_columns_1 = ['prefix', 'query_prediction', 'tag', 'title', 'label']
# kmeans_df = kmeans_df.drop(columns=drop_columns_1)
text_df = text_df.drop(columns=drop_columns_1)

drop_columns_s = ['label', 'tag', 'complete_prefix', 'prefix_word_num', 'title_len', 'query_length', 'prob_sum', 'prob_mean']
stat_df = stat_df.drop(columns=drop_columns_s)


# drop_columns_n = ['complete_prefix_click', 'complete_prefix_count','complete_prefix_ctr', 'complete_prefix_title_click',
#                   'complete_prefix_title_count', 'complete_prefix_title_ctr',
#                   'complete_prefix_tag_click', 'complete_prefix_tag_count',
#                   'complete_prefix_tag_ctr']
# nunique_df = nunique_df.drop(columns=drop_columns_n)

train_df = utils.read_txt('../data_train.txt')
validate_df = utils.read_txt('../data_vali.txt')
test_df = utils.read_txt('../data_test.txt',is_label=False)

train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]

import gc
df = pd.concat([ctr_df, prefix_df, query_df, text_df, stat_df, ctr_df_2,com_prefix_df], axis=1)
del ctr_df, prefix_df, query_df, text_df, stat_df, ctr_df_2,com_prefix_df
gc.collect()

for col in df.columns:
    print(col,':',df[col].isnull().sum())

with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','prefix_title_cvstat_cvr',
                'prefix_tag_cvstat_cvr','title_tag_cvstat_cvr','prefix_title_tag_cvstat_cvr',
               'complete_prefix_cvstat_cvr', 'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr','complete_prefix_title_tag_cvstat_cvr','cate_title_ctr',
               'cate_prefix_cate_title_ctr','cate_prefix_tag_ctr','cate_title_tag_ctr', 'prefix_num_title_num_tag_ctr']
    for col in features:
        df[col].fillna(df[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot',
                'title_tag_cvstat_hot','prefix_title_tag_cvstat_hot','complete_prefix_cvstat_hot',
       'complete_prefix_title_cvstat_hot','complete_prefix_tag_cvstat_hot','complete_prefix_title_tag_cvstat_hot',
    'prefix_w2v','max_similar','mean_similar','weight_similar','prefix_w2v_c','cate_prefix_click', 'cate_prefix_count', 'cate_prefix_ctr',
       'cate_title_click', 'cate_title_count', 
       'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
        'cate_prefix_tag_click','cate_prefix_tag_count', 
       'cate_title_tag_click', 'cate_title_tag_count',
        'prefix_num_title_num_tag_click',
       'prefix_num_title_num_tag_count']
    for col in features:
        df[col].fillna(0.0, inplace=True)
        
#df = df.fillna(0)

for col in df.columns:
    print(col,':',df[col].isnull().sum())

drop_columns_2 = ['prefix', 'query_prediction', 'title', 'cut_prefix', 'cut_title', 'cut_query_prediction', 'words']
#'prefix_num', 'title_num'
df = df.drop(columns=drop_columns_2)
df = pd.get_dummies(df, columns=['tag'])

train_data = df[:train_df_length]
train_data["label"] = train_data["label"].apply(int)

validate_data = df[train_df_length:train_df_length + validate_df_length]
validate_data["label"] = validate_data["label"].apply(int)

test_data = df[train_df_length + validate_df_length:]
test_data = test_data.drop(columns=["label"])


train_data_name = "train_pq.csv"
validate_data_name = "validate_pq.csv"
test_data_name = "test_pq.csv"

train_data.to_csv(train_data_name, index=False)
validate_data.to_csv(validate_data_name, index=False)
test_data.to_csv(test_data_name, index=False)