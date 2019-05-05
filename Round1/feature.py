# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:43:58 2018

@author: PQ
"""

import pandas as pd
import json
import time
import jieba
import re
from gensim.models import Word2Vec
import warnings
from gensim.models.keyedvectors import KeyedVectors
import os
import numpy as np



warnings.filterwarnings('ignore')


w2v_model_name = "w2v.bin"
if not os.path.exists(w2v_model_name):
    build_model(fname="train")
w2v_model = KeyedVectors.load_word2vec_format("w2v.bin", binary=True, unicode_errors="ignore")


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



class Processing(object):
    @staticmethod
    def _get_data(name):
        if name == "test":
            name = "test_A_20180929"
            columns = ['prefix', 'query_prediction', 'title', 'tag']
        else:
            columns = ['prefix', 'query_prediction', 'title', 'tag', 'label']

        data_name = "oppo_round1_{}.txt".format(name)
        df = pd.read_table(data_name, names=columns, header=None, encoding="utf-8").astype(str)
        return df

    @staticmethod
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
            ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 3)  # 平滑

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
                ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 3)  # 平滑

                df = pd.merge(df, ctr_df, how="left", on=group_column)

        return df

    def get_processing(self):
        train_df = self._get_data(name="train_20180929")
        train_df = train_df[train_df['label'] != '音乐']
        validate_df = self._get_data(name="vali_20180929")
        test_df = self._get_data(name="test")

        train_df_length = train_df.shape[0]
        validate_df_length = validate_df.shape[0]
        df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True)

        # make query prediction to json
        df["query_prediction"] = df["query_prediction"].apply(json.loads)

        # clearn prefix and title
        df["prefix"] = df["prefix"].apply(char_cleaner)
        df["title"] = df["title"].apply(char_cleaner)

        df = self._get_ctr_df(df, train_df_length)

        prefix_processing = PrefixProcessing()
        prefix_df = prefix_processing.get_prefix_df(df)

        query_processing = QueryProcessing()
        query_df = query_processing.get_query_df(df)

        df = pd.concat([df, prefix_df, query_df], axis=1)

        drop_columns = ['prefix', 'query_prediction', 'title', 'tag']
        df = df.drop(columns=drop_columns)

        train_data = df[:train_df_length]
        train_data["label"] = train_data["label"].apply(int)

        validate_data = df[train_df_length:train_df_length + validate_df_length]
        validate_data["label"] = validate_data["label"].apply(int)

        test_data = df[train_df_length + validate_df_length:]
        test_data = test_data.drop(columns=["label"])

        train_data_name = "train.csv"
        validate_data_name = "validate.csv"
        test_data_name = "test.csv"

        train_data.to_csv(train_data_name, index=False)
        validate_data.to_csv(validate_data_name, index=False)
        test_data.to_csv(test_data_name, index=False)
        

if __name__ == "__main__":
    t0 = time.time()
    Processing().get_processing()
    print(time.time() - t0)