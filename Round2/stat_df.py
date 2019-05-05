# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:23:54 2018

@author: PQ
"""

import json
import time
import warnings
from operator import itemgetter
import utils
import re

import jieba
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

class QueryProcessing(object):
    @staticmethod
    def _get_query_dict(item):
        item_dict = dict()

        query_predict = item["query_prediction"]

        if not query_predict:
            item_dict["query_length"] = 0
            item_dict["prob_sum"] = None
            item_dict["prob_max"] = None
            item_dict["prob_mean"] = None
            item_dict["prob_std"] = None
            item_dict["prob_min"] = None
            return item_dict

        prob_list = list()
        for _, prob in query_predict.items():
            prob = float(prob)
            prob_list.append(prob)

        item_dict["query_length"] = len(prob_list)
        item_dict["prob_sum"] = np.sum(prob_list)
        item_dict["prob_max"] = np.max(prob_list)
        item_dict["prob_mean"] = np.mean(prob_list)
        item_dict["prob_std"] = np.std(prob_list)
        item_dict["prob_min"] = np.min(prob_list)

        return item_dict

    def get_query_df(self, df):
        query_df = pd.DataFrame()

        query_df["item_dict"] = df.apply(self._get_query_dict, axis=1)
        query_df["query_length"] = query_df["item_dict"].apply(lambda item: item.get("query_length"))
        query_df["prob_sum"] = query_df["item_dict"].apply(lambda item: item.get("prob_sum"))
        query_df["prob_max"] = query_df["item_dict"].apply(lambda item: item.get("prob_max"))
        query_df["prob_mean"] = query_df["item_dict"].apply(lambda item: item.get("prob_mean"))
        query_df["prob_std"] = query_df["item_dict"].apply(lambda item: item.get("prob_std"))
        query_df["prob_min"] = query_df["item_dict"].apply(lambda item: item.get("prob_min"))
        query_df = query_df.drop(columns=["item_dict"])

        return query_df


    
class Processing(object):

    @staticmethod
    def _loads(item):
        try:
            return json.loads(item)
        except (json.JSONDecodeError, TypeError):
            return json.loads("{}")


    @staticmethod
    def _get_complete_prefix(item):
        prefix = item['prefix']
        query_prediction = item['query_prediction']

        if not query_prediction:
            return prefix

        predict_word_dict = dict()
        prefix = str(prefix)

        for query_item, query_ratio in query_prediction.items():
            query_item_cut = jieba.lcut(query_item)
            item_word = ""
            for item in query_item_cut:
                if prefix not in item_word:
                    item_word += item
                else:
                    if item_word not in predict_word_dict.keys():
                        predict_word_dict[item_word] = 0.0
                    predict_word_dict[item_word] += float(query_ratio)

        if not predict_word_dict:
            return prefix

        predict_word_dict = sorted(predict_word_dict.items(), key=itemgetter(1), reverse=True)
        complete_prefix = predict_word_dict[0][0]
        return complete_prefix

    @staticmethod
    def _get_max_query_ratio(item):
        query_prediction = item['query_prediction']
        title = item['title']

        if not query_prediction:
            return 0

        for query_wrod, ratio in query_prediction.items():
            if title == query_wrod:
                if float(ratio) > 0.1:
                    return 1

        return 0

    @staticmethod
    def _get_word_length(item):
        item = str(item)

        word_cut = jieba.lcut(item)
        length = len(word_cut)
        return length

    @staticmethod
    def _get_small_query_num(item):
        small_query_num = 0

        for _, ratio in item.items():
            if float(ratio) <= 0.08:
                small_query_num += 1

        return small_query_num

    def _get_length_df(self, df):
        df = df.copy()

        columns = ['query_prediction', 'prefix', 'title']
        length_df = df[columns]

        length_df['max_query_ratio'] = length_df.apply(self._get_max_query_ratio, axis=1)
        length_df['prefix_word_num'] = length_df['prefix'].apply(self._get_word_length)
        length_df['title_word_num'] = length_df['title'].apply(self._get_word_length)
        length_df['title_len'] = length_df['title'].apply(len)
        length_df['small_query_num'] = length_df['query_prediction'].apply(self._get_small_query_num)

        length_df = length_df.drop(columns=columns)
        return length_df

    def get_processing(self):
        train_df = utils.read_txt('../../data_train.txt')
        validate_df = utils.read_txt('../../data_vali.txt')
        test_df = utils.read_txt('../../data_test.txt',is_label=False)
        print('finish load data!')

        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True, sort=False)
    
        df["prefix"] = df["prefix"].apply(text_clean)
        df["title"] = df["title"].apply(text_clean)
        # make query prediction to json
        
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
        
        df["query_prediction"] = df["query_prediction"].apply(self._loads)
        
        # complete prefix
        df['complete_prefix'] = df[['prefix', 'query_prediction']].apply(self._get_complete_prefix, axis=1)
        print('finish get complete prefix!')

        length_df = self._get_length_df(df)
        print('finish get length df!')

        # clearn prefix and title

        df["complete_prefix"] = df["complete_prefix"].apply(text_clean)
        
        print('finish clearn columns!')

        query_processing = QueryProcessing()
        query_df = query_processing.get_query_df(df)
        print('finish get query df!')

        df = pd.concat([df, length_df, query_df], axis=1)
        print('finish combine all df!')

        drop_columns = ['prefix', 'query_prediction', 'title']
        df = df.drop(columns=drop_columns)
        
        df.to_csv('stat_df_x.csv', index=False)


if __name__ == "__main__":
    t0 = time.time()
    processing = Processing()
    processing.get_processing()
    print(time.time() - t0)