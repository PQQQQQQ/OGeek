import pandas as pd
import numpy as np
import json
import jieba
import time
import utils
import re
from gensim.models.keyedvectors import KeyedVectors
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib

warnings.filterwarnings('ignore')
t0 = time.time()


import unicodedata
from itertools import groupby
import re


w2v_model = KeyedVectors.load_word2vec_format('~/jupyter/Demo/DataSets/sgns.merge.word', binary=False, unicode_errors='ignore')


def text_clean(text,patterns=[]):
    patterns = [('','')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    """ Simple text clean up process"""
    clean = text.lower()
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return clean

train_df = utils.read_txt('../../data_train.txt')
validate_df = utils.read_txt('../../data_vali.txt')
test_df = utils.read_txt('../../data_test.txt',is_label=False)

train_df_length = train_df.shape[0]
validate_df_length = validate_df.shape[0]
df = pd.concat([train_df, validate_df, test_df], axis=0, ignore_index=True)

# make query prediction to json
# df["query_prediction"] = df["query_prediction"].apply(json.loads)

# df['query_prediction'] = df[df.query_prediction!='']['query_prediction'].apply(json.loads)

def _loads(item):
    try:
        return json.loads(item)
    except (json.JSONDecodeError, TypeError):
        return json.loads("{}")
    
# clearn prefix and title
df["prefix"] = df["prefix"].apply(text_clean)
df["title"] = df["title"].apply(text_clean)

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

df["prefix_w2v_2"] = df.apply(_get_prefix_w2v, axis=1)

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

df['complete_prefix'] = df[['prefix', 'query_prediction']].apply(_get_complete_prefix, axis=1)

def _get_comp_prefix_w2v(item):
    prefix = item["complete_prefix"]
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

df["prefix_w2v_c_2"] = df.apply(_get_comp_prefix_w2v, axis=1)

def _get_w2v_similar(item):
        item_dict = dict()

        query_predict = item["query_prediction"]
        title = item["title"]

        if not query_predict:
            item_dict["max_similar_2"] = None
            item_dict["mean_similar_2"] = None
            item_dict["weight_similar_2"] = None
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

        item_dict["max_similar_2"] = max_similar
        item_dict["mean_similar_2"] = mean_similar
        item_dict["weight_similar_2"] = weight_similar
        return item_dict

def get_query_df(df):
    query_df = pd.DataFrame()

    query_df["item_dict"] = df.apply(_get_w2v_similar, axis=1)
    query_df["max_similar_2"] = query_df["item_dict"].apply(lambda item: item.get("max_similar"))
    query_df["mean_similar_2"] = query_df["item_dict"].apply(lambda item: item.get("mean_similar"))
    query_df["weight_similar_2"] = query_df["item_dict"].apply(lambda item: item.get("weight_similar"))
    query_df = query_df.drop(columns=["item_dict"])

    return query_df

query_df = get_query_df(df)

df = pd.concat([df,query_df],axis=1)

f=open('../../xzy/stopwords.txt','r')
stopwords=[]
for line in f.readlines():
    line = line.strip() # 去掉首尾的空白
    stopwords.append(line)
f.close()

def lcut_word(text):
    str_mapper = str.maketrans("","",string.punctuation)
    seg_list=jieba.lcut(text.translate(str_mapper))
    #word_list = [item for item in seg_list]
    word_list=list(set(seg_list) - set(stopwords))
    return word_list

# vali_df = utils.read_txt('../../data_vali.txt')
# train_df = utils.read_txt('../../data_train.txt')
# test_df = utils.read_txt('../../data_test.txt',is_label=False)

# df = pd.concat([train_df, vali_df,test_df], axis=0, ignore_index=True)

df['cutprefix']=df['prefix'].apply(lcut_word)
df['cuttitle']=df['title'].apply(lcut_word)

def word_vec(word):
    if word == 'null':
        return np.zeros(100,)
    elif not word:
        return np.zeros(100,)
    else:
        try:
            similar_list = w2v_model[word]
            return matutils.unitvec(np.array(similar_list).mean(axis=0))
        except KeyError:
            return np.zeros(100,)

df['cutprefix_vec']=df['cutprefix'].apply(word_vec)

cutprefix_vec=np.stack(df['cutprefix_vec'],axis=0)

kmeans = MiniBatchKMeans(n_clusters=20, reassignment_ratio=0.001)
preds = kmeans.fit_predict(cutprefix_vec)
# joblib.dump(kmeans , 'kmeansprefix_w2v.pkl')
df['cate_prefix_2']=preds

df = df.drop(columns = ['cutprefix','cutprefix_vec'])

df['cuttitle_vec']=df['cuttitle'].apply(word_vec)

cuttitle_vec=np.stack(df['cuttitle_vec'],axis=0)

preds = kmeans.fit_predict(cuttitle_vec)

df['cate_title_2']=preds

df = df.drop(columns = ['cuttitle','cuttitle_vec'])


def _get_ctr_df(df, train_df_length):
    df = df.copy()
        
    train_df = df[:train_df_length]
    train_df['label'] = train_df['label'].apply(lambda x : int(x))
                    
    # kmeans features
    labels_columns_1 = ["cate_prefix_2", "cate_title_2"]
    for column in labels_columns_1:
        click_column = "{column}_click".format(column=column)
        count_column = "{column}_count".format(column=column)
        ctr_column = "{column}_ctr".format(column=column)

        agg_dict = {click_column: "sum", count_column: "count"}
        ctr_df = train_df.groupby(column, as_index=False)["label"].agg(agg_dict)
        ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

        df = pd.merge(df, ctr_df, how="left", on=column)
    
    labels_columns_2 = ["cate_prefix_2", "cate_title_2", "tag"]
    for idx, column1 in enumerate(labels_columns_2):
        for column2 in labels_columns_2[idx + 1:]:
            group_column = [column1, column2]
            click_column = "{column}_click".format(column="_".join(group_column))
            count_column = "{column}_count".format(column="_".join(group_column))
            ctr_column = "{column}_ctr".format(column="_".join(group_column))

            agg_dict = {click_column: "sum", count_column: "count"}
            ctr_df = train_df.groupby(group_column, as_index=False)["label"].agg(agg_dict)
            ctr_df[ctr_column] = ctr_df[click_column] / (ctr_df[count_column] + 5)  # 平滑

            df = pd.merge(df, ctr_df, how="left", on=group_column)
            
    return df

df = _get_ctr_df(df, train_df_length)

df.to_csv('word2_feature.csv',index=False)
