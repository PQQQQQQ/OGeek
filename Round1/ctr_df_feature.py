# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 16:13:53 2018

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


    # labels_columns_1 = ["prefix_kmeans", "title_kmeans"]
    # for column in labels_columns_1:
    #     click_column = "{column}_click".format(column=column)
    #     count_column = "{column}_count".format(column=column)
    #     ctr_column = "{column}_ctr".format(column=column)

    #     agg_dict = {click_column: "sum", count_column: "count"}
    #     ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
    #     ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

    #     df = pd.merge(df, ctr_df, how="left", on=column)

    # # combine features
    # labels_columns_2 = ["prefix_kmeans", "title_kmeans", "tag"]
    # for idx, column1 in enumerate(labels_columns_2):
    #     for column2 in labels_columns_2[idx + 1:]:
    #         group_column = [column1, column2]
    #         click_column = "{column}_click".format(column="_".join(group_column))
    #         count_column = "{column}_count".format(column="_".join(group_column))
    #         ctr_column = "{column}_ctr".format(column="_".join(group_column))

    #         agg_dict = {click_column: "sum", count_column: "count"}
    #         ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
    #         ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

    #         df = pd.merge(df, ctr_df, how="left", on=group_column)

# 线下有提升，线上没有效果(可能过拟合)
        # prefix_num title_num features
    # labels_columns_3 = ["prefix_num", "title_num"]
    # for column in labels_columns_3:
    #     click_column = "{column}_click".format(column=column)
    #     count_column = "{column}_count".format(column=column)
    #     ctr_column = "{column}_ctr".format(column=column)

    #     agg_dict = {click_column: "sum", count_column: "count"}
    #     ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
    #     ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

    #     df = pd.merge(df, ctr_df, how="left", on=column)

    # labels_columns_4 = ["prefix_num", "title_num", "tag"]
    # for idx, column1 in enumerate(labels_columns_4):
    #     for column2 in labels_columns_2[idx + 1:]:
    #         group_column = [column1, column2]
    #         click_column = "{column}_click".format(column="_".join(group_column))
    #         count_column = "{column}_count".format(column="_".join(group_column))
    #         ctr_column = "{column}_ctr".format(column="_".join(group_column))

    #         agg_dict = {click_column: "sum", count_column: "count"}
    #         ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
    #         ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

    #         df = pd.merge(df, ctr_df, how="left", on=group_column)

        
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


    # group_column_2 = ['prefix_kmeans', 'title_kmeans', 'tag']
    # click_column_3 = "{column}_click".format(column="_".join(group_column_2))
    # count_column_3 = "{column}_count".format(column="_".join(group_column_2))
    # ctr_column_3 = "{column}_ctr".format(column="_".join(group_column_2))
    # agg_dict_3 = {click_column_3: "sum", count_column_3: "count"}

    # temp3 = train_df.groupby(['prefix_kmeans','title_kmeans','tag'], as_index=False)['label'].agg(agg_dict_3)
    # temp3[ctr_column_3] = temp3[click_column_3] / (temp3[count_column_3] + 5)  # 平滑
    # df = pd.merge(df, temp3, how="left", on=group_column_2)

    return df

def _get_data(name):
    if name == "test":
        name = "test_B_20181106"
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


df['prefix_num'] = df['prefix'].apply(lambda x:len(x))
df['title_num'] = df['title'].apply(lambda x:len(x))


# df["query_prediction"] = df["query_prediction"].apply(json.loads)

# # clearn prefix and title
# df["prefix"] = df["prefix"].apply(char_cleaner)
# df["title"] = df["title"].apply(char_cleaner)


ctr_df = _get_ctr_df(df, train_df_length)

ctr_df.to_csv('ctr_df_n.csv', index=False)