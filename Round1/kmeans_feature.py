# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:18:40 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import json
import jieba
import re
import os
import time
from gensim import matutils
from sklearn.cluster import MiniBatchKMeans
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
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



def _to_csv(df, col, size):
        file_name = '{col}_w2v.csv'.format(col=col)
        file_path = file_name
        if os.path.exists(file_path):
            os.remove(file_path)

        columns = ['{}_w2v_{}'.format(col, i) for i in range(size)]
        none_index_set = set()

        with open(file_path, 'a', encoding='utf-8') as f:
            # write columns
            f.write(','.join(columns) + '\n')

            for idx, item in tqdm(df[col].items()):
                if item == 'null':
                    item_list = [''] * size
                    none_index_set.add(idx)
                elif not item:
                    item_list = [''] * size
                    none_index_set.add(idx)
                else:
                    seg_cut = jieba.lcut(item)
                    seg_cut = char_list_cheaner(seg_cut)

                    w2v_array = list()
                    for word in seg_cut:
                        try:
                            similar_list = w2v_model[word]
                            w2v_array.append(similar_list)
                        except KeyError:
                            pass

                    if not w2v_array:
                        item_list = [''] * size
                        none_index_set.add(idx)
                    else:
                        item_list = matutils.unitvec(np.array(w2v_array).mean(axis=0))

                f.write(','.join(map(str, item_list)) + '\n')

        return none_index_set

def _get_w2v_df(df, col, size=500):
    none_index_set = _to_csv(df, col, size)

    file_name = '{col}_w2v.csv'.format(col=col)
    file_path = file_name

    w2v_df = pd.read_csv(file_path, header=0)
    w2v_df['help_index'] = w2v_df.index
    w2v_df['help_flag'] = w2v_df['help_index'].apply(lambda _item: 0 if _item in none_index_set else 1)

    return w2v_df

def _get_kmeans_dict(df, size=20):
    df = df.copy()
    df = df[df['help_flag'] == 1]
    help_index = df['help_index'].tolist()

    df = df.drop(columns=['help_index', 'help_flag'])

    kmeans = MiniBatchKMeans(n_clusters=size, reassignment_ratio=0.001)
    preds = kmeans.fit_predict(df)

    kmeans_dict = dict(zip(help_index, preds))
    return kmeans_dict

def _mapping_kmeans(item, mapping_dict):
    return mapping_dict.get(item, -1)


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



prefix_w2v_df = _get_w2v_df(df, col='prefix')
title_w2v_df = _get_w2v_df(df, col='title')

prefix_kmeans_dict = _get_kmeans_dict(prefix_w2v_df)
title_kmeans_dict = _get_kmeans_dict(title_w2v_df)
df['prefix_kmeans'] = prefix_w2v_df['help_index'].apply(_mapping_kmeans, args=(prefix_kmeans_dict,))
df['title_kmeans'] = title_w2v_df['help_index'].apply(_mapping_kmeans, args=(title_kmeans_dict,))


df.to_csv('kmeans_df.csv', index=False)