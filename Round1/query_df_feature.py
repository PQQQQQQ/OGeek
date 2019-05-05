# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:03:58 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import json
import jieba
import re
from gensim.models.keyedvectors import KeyedVectors
import warnings


warnings.filterwarnings('ignore')


w2v_model = KeyedVectors.load_word2vec_format("w2v.bin", binary=True, unicode_errors="ignore")


def char_cleaner(char):
    if not isinstance(char, str):
        char = "null"

    pattern = re.compile("[^a-zA-Z\u4E00-\u9FA5 ]")
    char = re.sub(pattern, "", char)
    char = char.lower()
    return char


def char_list_cheaner(char_list, stop_words=None):
    new_char_list = list()
    for char in char_list:
        if len(char) <= 1:
            continue
        if stop_words and char in stop_words:
            continue
        new_char_list.append(char)

    return new_char_list



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
    

def _get_data(name):
    if name == "test":
        name = "test_A_20180929"
        columns = ['prefix', 'query_prediction', 'title', 'tag']
    else:
        columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']

    data_name = "oppo_round1_{}.txt".format(name)
    df = pd.read_table(data_name, names=columns, header=None, encoding="utf-8").astype(str)
    return df
    

train_df = _get_data(name="train_20180929")
train_df = train_df[train_df['label'] != '音乐']
validate_df = _get_data(name="vali_20180929")
test_df = _get_data(name="test")

train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]
df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True)

# make query prediction to json
df["query_prediction"] = df["query_prediction"].apply(json.loads)

# clearn prefix and title
df["prefix"] = df["prefix"].apply(char_cleaner)
df["title"] = df["title"].apply(char_cleaner)


query_processing = QueryProcessing()
query_df = query_processing.get_query_df(df)

query_df.to_csv('query_df.csv', index=False)