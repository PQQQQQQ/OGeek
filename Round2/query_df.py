# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:38:17 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import json
import jieba
import time
import utils
import re
from gensim.models.keyedvectors import KeyedVectors
import warnings
import gc

warnings.filterwarnings('ignore')
t0 = time.time()


# w2v_model = KeyedVectors.load_word2vec_format("../w2v_100.bin", binary=True, unicode_errors="ignore")
#w2v_model = KeyedVectors.load_word2vec_format("../../xty/resources/w2v_new100.bin", binary=True, unicode_errors="ignore")

w2v_mdoel = KeyedVectors.load_word2vec_format('~/jupyter/Demo/DataSets/sgns.merge.word', binary=False, unicode_errors='ignore')

# def char_cleaner(char):
#     if not isinstance(char, str):
#         char = "null"

#     pattern = re.compile("[^0-9a-zA-Z\u4E00-\u9FA5 ]")
#     char = re.sub(pattern, "", char)
#     char = char.lower()
#     return char


def char_list_cheaner(char_list, stop_words=None):
    new_char_list = list()
    for char in char_list:
        if len(char) <= 1:
            continue
        if stop_words and char in stop_words:
            continue
        new_char_list.append(char)

    return new_char_list

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


class QueryProcessing(object):

    @staticmethod
    def _get_w2v_similar(item):
        item_dict = dict()

        query_predict = item["query_prediction"]
        title = item["title"]

        if not query_predict:
            item_dict["max_similar"] = None
            item_dict["mean_similar"] = None
            item_dict["weight_similar"] = None
            return item_dict

        similar_list = list()
        weight_similar_list = list()

        title_cut = list(jieba.cut(title))
        title_cut = char_list_cheaner(title_cut)
        for key, value in query_predict.items():
            query_cut = list(jieba.cut(key))
            query_cut = char_list_cheaner(query_cut)

            try:
                w2v_similar = w2v_model.n_similarity(query_cut, title_cut)
            except (KeyError, ZeroDivisionError):
                w2v_similar = np.nan

            similar_list.append(w2v_similar)
            weight_w2v_similar = w2v_similar * float(value)
            weight_similar_list.append(weight_w2v_similar)

        max_similar = np.nanmax(similar_list)
        mean_similar = np.nanmean(similar_list)
        weight_similar = np.nansum(weight_similar_list)

        item_dict["max_similar"] = max_similar
        item_dict["mean_similar"] = mean_similar
        item_dict["weight_similar"] = weight_similar
        return item_dict

    def get_query_df(self, df):
        query_df = pd.DataFrame()

        query_df["item_dict"] = df.apply(self._get_w2v_similar, axis=1)
        query_df["max_similar"] = query_df["item_dict"].apply(lambda item: item.get("max_similar"))
        query_df["mean_similar"] = query_df["item_dict"].apply(lambda item: item.get("mean_similar"))
        query_df["weight_similar"] = query_df["item_dict"].apply(lambda item: item.get("weight_similar"))
        query_df = query_df.drop(columns=["item_dict"])

        return query_df   


train_df = utils.read_txt('../../data_train.txt')
validate_df = utils.read_txt('../../data_vali.txt')
test_df = utils.read_txt('../../data_test.txt',is_label=False)


train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]
df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True)

del train_df,validate_df,test_df
gc.collect()
# make query prediction to json
# df["query_prediction"] = df["query_prediction"].apply(json.loads)

def _loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")
    



# clearn prefix and title
df["prefix"] = df["prefix"].apply(text_clean)
df["title"] = df["title"].apply(text_clean)

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

df["query_prediction"] = df["query_prediction"].apply(_loads)

query_processing = QueryProcessing()
query_df = query_processing.get_query_df(df)

query_df.to_csv('query_df_pq_x.csv', index=False)

print(time.time() - t0)