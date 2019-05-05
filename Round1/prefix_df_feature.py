# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 15:11:52 2018

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


class PrefixProcessing(object):
    @staticmethod
    def _is_in_title(item):
        prefix = item["prefix"]
        title = item["title"]

        if not isinstance(prefix, str):
            prefix = "null"

        if prefix in title:
            return 1
        return 0

    @staticmethod
    def _levenshtein_distance(item):
        str1 = item["prefix"]
        str2 = item["title"]

        if not isinstance(str1, str):
            str1 = "null"

        x_size = len(str1) + 1
        y_size = len(str2) + 1

        matrix = np.zeros((x_size, y_size), dtype=np.int_)

        for x in range(x_size):
            matrix[x, 0] = x

        for y in range(y_size):
            matrix[0, y] = y

        for x in range(1, x_size):
            for y in range(1, y_size):
                if str1[x - 1] == str2[y - 1]:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
                else:
                    matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)

        return matrix[x_size - 1, y_size - 1]

    @staticmethod
    def _distince_rate(item):
        str1 = item["prefix"]
        str2 = item["title"]
        leven_distance = item["leven_distance"]

        if not isinstance(str1, str):
            str1 = "null"

        length = max(len(str1), len(str2))

        return leven_distance / (length + 5)  # 平滑

    @staticmethod
    def _get_prefix_w2v(item):
        prefix = item["prefix"]
        title = item["title"]
        if not isinstance(prefix, str):
            prefix = "null"

        prefix_cut = list(jieba.cut(prefix))
        title_cut = list(jieba.cut(title))

        prefix_cut = char_list_cheaner(prefix_cut)
        title_cut = char_list_cheaner(title_cut)

        try:
            w2v_similar = w2v_model.n_similarity(prefix_cut, title_cut)
        except (KeyError, ZeroDivisionError):
            w2v_similar = None

        return w2v_similar

    def get_prefix_df(self, df):
        prefix_df = pd.DataFrame()

        prefix_df[["prefix", "title"]] = df[["prefix", "title"]]
        prefix_df["is_in_title"] = prefix_df.apply(self._is_in_title, axis=1)
        prefix_df["leven_distance"] = prefix_df.apply(self._levenshtein_distance, axis=1)
        prefix_df["distance_rate"] = prefix_df.apply(self._distince_rate, axis=1)
        prefix_df["prefix_w2v"] = prefix_df.apply(self._get_prefix_w2v, axis=1)
        return prefix_df


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


prefix_processing = PrefixProcessing()
prefix_df = prefix_processing.get_prefix_df(df)

prefix_df.to_csv('prefix_df.csv', index=False)