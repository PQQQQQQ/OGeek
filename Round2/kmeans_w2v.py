import pandas as pd
import numpy as np
import utils
import jieba
import string
from gensim import matutils
import time
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
start_time=time.time()
w2v_model = KeyedVectors.load_word2vec_format("../w2v_100.bin", binary=True, unicode_errors="ignore")
# 停用词的载入
f=open('stopwords.txt','r')
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

vali_df = utils.read_txt('../data_vali.txt')
train_df = utils.read_txt('../data_train.txt')
test_df = utils.read_txt('../data_test.txt',is_label=False)

df = pd.concat([train_df, vali_df,test_df], axis=0, ignore_index=True)

def lcut_word(text):
    str_mapper = str.maketrans("","",string.punctuation)
    seg_list=jieba.lcut(text.translate(str_mapper))
    #word_list = [item for item in seg_list]
    word_list=list(set(seg_list) - set(stopwords))
    return word_list
df['cutprefix']=df['prefix'].apply(lcut_word)

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
joblib.dump(kmeans , 'kmeansprefix_w2v.pkl')
df['cate_prefix']=preds

df.to_csv('kmeansprefix_w2v.csv',index=None)
end_time=time.time()
print('time:',(end_time-start_time)/60)