# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:39:12 2018

@author: PQ
"""

import pandas as pd
import json
import time
import jieba
import re
from gensim.models import Word2Vec


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


def get_sentence(fname="train"):
    fname = "oppo_round1_{fname}_20180929.txt".format(fname=fname)
    
    with open('oppo_round1_train_20180929.txt', "r", encoding="utf-8") as f:
        line = f.readline()

        while line:
            line_arr = line.split("\t")

            query_prediction = line_arr[1]
            sentences = json.loads(query_prediction)
            for sentence in sentences:
                yield char_cleaner(sentence)

            title = line_arr[2]
            yield char_cleaner(title)

            line = f.readline()
            
class MySentence(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for sentence in get_sentence(self.fname):
            seg_list = jieba.cut(sentence)
            seg_list = list(seg_list)
            seg_list = char_list_cheaner(seg_list)
            if seg_list:
                yield seg_list
                

def build_model(fname):
    sentences = MySentence(fname)
    model_name = "w2v.bin"
    my_model = Word2Vec(sentences, size=500, window=5, sg=1, hs=1, min_count=2, workers=10)

    my_model.wv.save_word2vec_format(model_name, binary=True)
    

if __name__ == "__main__":
    t0 = time.time()
    build_model(fname="train")
    print(time.time() - t0)