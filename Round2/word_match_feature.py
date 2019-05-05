import pandas as pd
import numpy as np
import string
import jieba
import utils
import json
import time
import warnings


warnings.filterwarnings('ignore')
t0 = time.time()


df_valid = utils.read_txt('../data_vali.txt')
df_train = utils.read_txt('../data_train.txt')
df_test = utils.read_txt('../data_testb.txt',is_label=False)


df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)

df_test['label']=-1
        
df_data=pd.concat([df_train, df_valid, df_test]).reset_index(drop=True)


def word_match_share(row):
    p_words = list(jieba.cut(row['prefix']))
    t_words = list(jieba.cut(row['title']))

    if len(p_words) == 0 or len(t_words) == 0:
        return 0.
    
    shared_words_in_p = [w for w in p_words if w in t_words]
    shared_words_in_t = [w for w in t_words if w in p_words]
    R = float(len(shared_words_in_p) + len(shared_words_in_t))/(len(p_words) + len(t_words))
    return R

def char_match_share(row):
    p_chars = row['prefix']
    t_chars = row['title']

    if len(p_chars) == 0 or len(t_chars) == 0:
        return 0.
    
    shared_chars_in_p = [w for w in p_chars if w in t_chars]
    shared_chars_in_t = [w for w in t_chars if w in p_chars]
    R = float(len(shared_chars_in_p) + len(shared_chars_in_t))/(len(p_chars) + len(t_chars))
    return R

def max_pred_char_match_share(row):
    R = 0
    if row['query_prediction']:
        p_chars, score = get_max_score_pred(eval(row['query_prediction']))
        t_chars = row['title']

        if len(p_chars) == 0 or len(t_chars) == 0:
            return 0.

        shared_chars_in_p = [w for w in p_chars if w in t_chars]
        shared_chars_in_t = [w for w in t_chars if w in p_chars]
        R = float(len(shared_chars_in_p) + len(shared_chars_in_t))/(len(p_chars) + len(t_chars))
    return R

df_data['word_match'] = df_data.apply(lambda x:word_match_share(x),axis=1)
df_data['char_match'] = df_data.apply(lambda x:char_match_share(x),axis=1)
df_data['max_pred_char_match'] = df_data.apply(lambda x:max_pred_char_match_share(x),axis=1)

df_data.to_csv('feature/wordmatch_df_b.csv',index=False)