# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:30:08 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import json
import jieba
import re
import os
import time
import utils
from gensim import matutils
from sklearn.cluster import MiniBatchKMeans
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings


warnings.filterwarnings('ignore')
import unicodedata
from itertools import groupby
import re
import gc


warnings.filterwarnings('ignore')
t0 = time.time()

# def text_clean(text,patterns=[]):
#     patterns = [('','')]
#     patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
#     """ Simple text clean up process"""
#     clean = text.lower()
#     for (pattern, repl) in patterns:
#         clean = re.sub(pattern, repl, clean)
#     return clean

w2v_model = KeyedVectors.load_word2vec_format("../w2v_b_100.bin", binary=True, unicode_errors="ignore")


def char_cleaner(char):
    if not isinstance(char, str):
        char = "null"

    pattern = re.compile("[^0-9a-zA-Z\u4E00-\u9FA5 ]")
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
    
    
def _get_w2v_df(df, col, size=100):
    none_index_set = _to_csv(df, col, size)

    file_name = '{col}_w2v.csv'.format(col=col)
    file_path = file_name

    w2v_df = pd.read_csv(file_path, header=0)
    w2v_df['help_index'] = w2v_df.index
    w2v_df['help_flag'] = w2v_df['help_index'].apply(lambda _item: 0 if _item in none_index_set else 1)

    return w2v_df


def _get_prefix_df(prefix_w2v_df, title_w2v_df, col_name):
    prefix_df = pd.DataFrame()

    remove_columns = ['help_index', 'help_flag']

    prefix_w2v_df = prefix_w2v_df.copy()
    prefix_w2v_df = prefix_w2v_df.drop(columns=remove_columns)

    title_w2v_df = title_w2v_df.copy()
    title_w2v_df = title_w2v_df.drop(columns=remove_columns)

    prefix_w2v_list = list()
    for idx, prefix in prefix_w2v_df.iterrows():
        if np.isnan(prefix[0]):
            prefix_w2v_list.append(None)
            continue

        title = title_w2v_df.loc[idx]
        if np.isnan(title[0]):
            prefix_w2v_list.append(None)
            continue

        similar = np.dot(prefix, title)
        prefix_w2v_list.append(similar)

    prefix_df[col_name] = prefix_w2v_list
    return prefix_df
    

#     @staticmethod
#     def _loads(item):
#         try:
#             return json.loads(item)
#         except (json.JSONDecodeError, TypeError):
#             return json.loads("{}")


def _get_pca_df(df, name, n_components=5):
    df = df.copy()

    remove_columns = ['help_flag', 'help_index']

    df_effective = df[df['help_flag'] == 1]
    df_invalid = df[df['help_flag'] == 0]

    df_effective = df_effective.drop(columns=remove_columns)
    df_invalid = df_invalid.drop(columns=remove_columns)

    pca_columns = ['{}_pca_{}'.format(name, i) for i in range(n_components)]

    pca = PCA(n_components=n_components)

    pca_data = pca.fit_transform(df_effective)
    pca_df = pd.DataFrame(pca_data, index=df_effective.index, columns=pca_columns)
    none_df = pd.DataFrame(index=df_invalid.index, columns=pca_columns)

    pca_df = pd.concat([pca_df, none_df], axis=0, ignore_index=False, sort=False)
    pca_df = pca_df.sort_index()

    return pca_df



        
train_df = utils.read_txt('../data_train.txt')
validate_df = utils.read_txt('../data_vali.txt')
test_df = utils.read_txt('../data_testb.txt',is_label=False)
print('finish load data!')

stat_df = pd.read_csv('feature/stat_df_b.csv')

drop_columns = ['max_query_ratio', 'prefix_word_num', 'title_word_num', 'title_len', 'small_query_num', 
                'query_length', 'prob_sum', 'prob_max', 'prob_mean', 'tag', 'label','prob_std', 'prob_min']
stat_df = stat_df.drop(columns=drop_columns)

train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]
df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)
df = pd.concat([df, stat_df], axis=1)

del train_df, validate_df, test_df
gc.collect()

#         # make query prediction to json
#         df["query_prediction"] = df["query_prediction"].apply(self._loads)

#         # complete prefix
#         df['complete_prefix'] = df[['prefix', 'query_prediction']].apply(self._get_complete_prefix, axis=1)

        # clearn prefix and title
df["prefix"] = df["prefix"].apply(char_cleaner)
df["title"] = df["title"].apply(char_cleaner)
df["complete_prefix"] = df["complete_prefix"].apply(char_cleaner)

w2v_df = df[['label']]

prefix_w2v_df = _get_w2v_df(df, col='prefix')
title_w2v_df = _get_w2v_df(df, col='title')
complete_prefix_w2v_df = _get_w2v_df(df, col='complete_prefix')
print('finish get prefix and title w2v df!')

prefix_pca_df = _get_pca_df(prefix_w2v_df, 'prefix')
title_pca_df = _get_pca_df(title_w2v_df, 'title')
complete_prefix_pca_df = _get_pca_df(complete_prefix_w2v_df, 'complete_prefix')
w2v_df = pd.concat([w2v_df, prefix_pca_df, title_pca_df, complete_prefix_pca_df], axis=1)

del prefix_pca_df, title_pca_df, complete_prefix_pca_df
gc.collect()
        

prefix_df = _get_prefix_df(prefix_w2v_df, title_w2v_df, 'prefix_w2v')
complete_prefix_df = _get_prefix_df(complete_prefix_w2v_df, title_w2v_df, 'complete_prefix_w2v')
print('finish get prefix df!')
w2v_df = pd.concat([w2v_df, prefix_df, complete_prefix_df], axis=1)

del prefix_df, prefix_w2v_df, title_w2v_df, complete_prefix_df
gc.collect()
        

w2v_df = w2v_df.drop(columns=['label'])
        
w2v_df.to_csv('w2v_df_b.csv', index=False)

print(time.time() - t0)