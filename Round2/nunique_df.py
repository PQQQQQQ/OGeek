# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:46:35 2018

@author: PQ
"""

import json
import time
import warnings
from operator import itemgetter
import utils
import re
import gc

import jieba
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
import scipy.special as special
import math
from math import log

warnings.filterwarnings('ignore')


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

# def char_cleaner(char):
#     if not isinstance(char, str):
#         char = "null"

#     pattern = re.compile("[^0-9a-zA-Z\u4E00-\u9FA5 ]")
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

    
class Processing(object):

#     @staticmethod
#     def _loads(item):
#         try:
#             return json.loads(item)
#         except (json.JSONDecodeError, TypeError):
#             return json.loads("{}")

    @staticmethod
    def _get_apriori_df(df, train_df_length, columns=None):
        df = df.copy()

        train_df = df[:train_df_length]
        train_df['label'] = train_df['label'].apply(lambda x : int(x))

        if columns is None:
            columns = [ 'complete_prefix']

        ctr_columns = columns.copy()
        ctr_columns.extend(['complete_prefix_title', 'complete_prefix_tag'])
        apriori_df = df[ctr_columns]

        # click count and ctr
        for idx, column in enumerate(ctr_columns):
            click_column = "{column}_click".format(column=column)
            count_column = "{column}_count".format(column=column)
            ctr_column = "{column}_ctr".format(column=column)

            agg_dict = {click_column: "sum", count_column: "count"}
            column_apriori_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
            column_apriori_df[ctr_column] = column_apriori_df[click_column] / (column_apriori_df[count_column] + 5)
            apriori_df = pd.merge(apriori_df, column_apriori_df, how='left', on=column)

        length = apriori_df.shape[0]
        all_columns = apriori_df.columns

        return apriori_df



    def get_processing(self):
        train_df = utils.read_txt('../../data_train.txt')
        validate_df = utils.read_txt('../../data_vali.txt')
        test_df = utils.read_txt('../../data_test.txt',is_label=False)
        print('finish load data!')

        stat_df = pd.read_csv('stat_df_x.csv')
        
        drop_columns = ['max_query_ratio', 'prefix_word_num', 'title_word_num', 'title_len', 'small_query_num', 
                        'query_length', 'prob_sum', 'prob_max', 'prob_mean', 'tag', 'label','prob_std', 'prob_min']
        stat_df = stat_df.drop(columns=drop_columns)
        
        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)
        df = pd.concat([df, stat_df], axis=1)


        # clearn prefix and title
        df["prefix"] = df["prefix"].apply(text_clean)
        df["title"] = df["title"].apply(text_clean)
        #df["complete_prefix"] = df["complete_prefix"].apply(text_clean)
        print('finish clearn columns!')

        # combine columns
        #df['prefix_title'] = df[['prefix', 'title']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        #df['prefix_tag'] = df[['prefix', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        df['complete_prefix_title'] = df[['complete_prefix', 'title']].apply(lambda item: '_'.join(map(str, item)),
                                                                             axis=1)
        df['complete_prefix_tag'] = df[['complete_prefix', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        
        #df['complete_prefix_title_tag'] = df[['complete_prefix', 'title','tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        
        #df['title_tag'] = df[['title', 'tag']].apply(lambda item: '_'.join(map(str, item)), axis=1)
        print('finish combine columns!')

        apriori_df = self._get_apriori_df(df, train_df_length)
        print('finish get apriori df!')



        df = pd.concat([df, apriori_df], axis=1)
        print('finish combine all df!')

        drop_columns = ['prefix', 'query_prediction', 'title', 'tag','label', 'complete_prefix','complete_prefix_title', 'complete_prefix_tag']
        df = df.drop(columns=drop_columns)
#         drop_columns = ['prefix', 'query_prediction', 'title', 'tag','label', 'complete_prefix', 'prefix_click', 'prefix_count', 'prefix_ctr','title_click', 'title_count', 'title_ctr',
#                   'tag_click', 'tag_count', 'tag_ctr', 'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr', 'prefix_tag_click',
#                   'prefix_tag_count', 'prefix_tag_ctr', 'title_tag_click', 'title_tag_count', 'title_tag_ctr','complete_prefix_title_tag', 'prefix_title', 'prefix_tag','complete_prefix_title', 'complete_prefix_tag', 'title_tag',
#        'complete_prefix_title_tag.1']
        
#         df = df.drop(columns=drop_columns)
        
        df.to_csv('nunique_df_pq_x.csv', index=False)




if __name__ == "__main__":
    t0 = time.time()
    processing = Processing()
    processing.get_processing()
    print(time.time() - t0)