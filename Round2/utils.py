#coding:utf-8
import pandas as pd, numpy as np
import os

import time
import pickle as pkl
from sklearn import preprocessing
from contextlib import contextmanager
from datetime import datetime

from gensim import corpora, models, similarities
import pandas as pd, numpy as np
import os
import scipy.sparse as sparse
import jieba
import logging
from sklearn.metrics import f1_score
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    print('--- [START %s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
def read_csv(fname,chunksize=10000,chunknum=-1,verbose=False):
    chunks = pd.read_csv(fname,chunksize=chunksize)
    df_chunks = []

    for idx,chunk in enumerate(chunks):
        if verbose==True:
            print("rocessing Chunk idx:",idx)
        df_chunks.append(chunk)
        if idx == chunknum-1:
            break   
    df = pd.concat(df_chunks).reset_index(drop=True)
    return df
def read_txt(fname,is_label=True):
    prefix = []
    query_prediction = []
    title = []
    tag = []
    label = []

    with open(fname,'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            # if len(line)==5:
            prefix.append(line[0])
            query_prediction.append(line[1])
            title.append(line[2])
            tag.append(line[3])
            if is_label:
                label.append(line[4])
    if is_label:
        return pd.DataFrame({'prefix':prefix,'query_prediction':query_prediction,'title':title,'tag':tag,'label':label})
    else:
        return pd.DataFrame({'prefix':prefix,'query_prediction':query_prediction,'title':title,'tag':tag})

#https://blog.csdn.net/u014595019/article/details/52433754/
def corpus2csr(corpus):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus:  #[[(),()]]
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    sparse_matrix = sparse.csr_matrix((data,(rows,cols))) # 稀疏向量
    return sparse_matrix
def clean_csr(csr_trn, csr_sub, min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    sub_min = {x for x in np.where(csr_sub.getnnz(axis=0) >= min_df)[0]}
    mask= [x for x in trn_min if x in sub_min]
    return csr_trn[:, mask], csr_sub[:, mask]



# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df

# 分词
def seg_words(contents):
    contents_segs = list()
    for content in contents:
        segs = jieba.lcut(content)
        contents_segs.append(" ".join(segs))
    return contents_segs
# https://blog.csdn.net/lv26230418/article/details/46356763
#-------------------------------------------------------------------------------    
def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    # format   = 'LINE %(lineno)-4d  %(levelname)-8s %(message)s',
                    format   = '%(message)s',
                    datefmt  = '%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'a');
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler();
    console.setLevel(logging.INFO);
    # set a format which is simpler for console use
    # formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s');
    formatter = logging.Formatter('%(message)s');
    # tell the handler to use this format
    console.setFormatter(formatter);
    logging.getLogger('').addHandler(console);
    return logging.getLogger('')
def macro_f1_score(y_trues,y_probs):
    return f1_score(y_trues, y_probs, average='macro')

def load_glove_txt(fname,all_words=[],encoding="utf-8"):
    glove = {}
    with open(fname, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(encoding)
            if len(all_words)>0:
                if (word in all_words):
                    nums=np.array(parts[1:], dtype=np.float32)
                    glove[word] = nums
            else:
                nums=np.array(parts[1:], dtype=np.float32)
                glove[word] = nums
    return glove
# net ------------------------------------
# https://github.com/pytorch/examples/blob/master/imagenet/main.py ###############
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr
