# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:09:39 2018

@author: PQ
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from scipy import sparse
import pandas as pd
import numpy as np
import datetime
import os


train = pd.read_table('oppo_round1_train_20180929.txt', 
                      names= ['prefix','query_prediction','title','tag','label'], header= None, encoding='utf-8').astype(str)

val = pd.read_table('oppo_round1_vali_20180929.txt', 
        names = ['prefix','query_prediction','title','tag','label'], header = None, encoding='utf-8').astype(str)

test = pd.read_table('oppo_round1_test_A_20180929.txt',
        names = ['prefix','query_prediction','title','tag'],header = None, encoding='utf-8').astype(str)

train = train[train['label'] != '音乐']

test['label'] = -1
train = pd.concat([train, val], axis=0, ignore_index=True)

train['label'] = train['label'].apply(lambda x: int(x))
test['label'] = test['label'].apply(lambda x: int(x))
items = ['prefix', 'title', 'tag']


for item in items:
    temp = train.groupby(item, as_index = False)['label'].agg({item+'_click':'sum', item+'_count':'count'})
    temp[item+'_ctr'] = temp[item+'_click']/(temp[item+'_count'])
    train = pd.merge(train, temp, on=item, how='left')
    test = pd.merge(test, temp, on=item, how='left')
    
    
for i in range(len(items)):
    for j in range(i+1, len(items)):
        item_g = [items[i], items[j]]
        temp = train.groupby(item_g, as_index=False)['label'].agg({'_'.join(item_g)+'_click': 'sum','_'.join(item_g)+'_count':'count'})
        temp['_'.join(item_g)+'_ctr'] = temp['_'.join(item_g)+'_click']/(temp['_'.join(item_g)+'_count']+3)
        train = pd.merge(train, temp, on=item_g, how='left')
        test = pd.merge(test, temp, on=item_g, how='left')
        
        
test = test.fillna(0)


train['tag'] = train['tag'].map(dict(zip(train['tag'].unique(), range(0, train['tag'].nunique()))))
test['tag'] = test['tag'].map(dict(zip(test['tag'].unique(), range(0, test['tag'].nunique()))))

data = pd.concat([train, test], axis=0, ignore_index=True)


predict = data[data.label == -1]
train_x = data[data.label != -1]
predict_x = predict.drop('label', axis=1)
train_y = data[data.label != -1].label.values
              

num_feature = ['prefix_click', 'prefix_count', 'prefix_ctr', 'title_click', 'title_count', 'title_ctr', 'tag_click', 'tag_count', 'tag_ctr',
               'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr', 'prefix_tag_click', 'prefix_tag_count', 'prefix_tag_ctr',
               'title_tag_click', 'title_tag_count', 'title_tag_ctr']

cate_feature = ['tag']


cv = CountVectorizer(min_df=20)

if os.path.exists('base_train_csr.npz') and False:
    print('load_csr---------')
    base_train_csr = sparse.load_npz('base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz('base_predict_csr.npz').tocsr().astype('bool')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')
    
    cv = CountVectorizer(min_df=20)
    for feature in ['prefix', 'query_prediction', 'title']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz('base_train_csr.npz', base_train_csr)
    sparse.save_npz('base_predict_csr.npz', base_predict_csr)

train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)


xx_logloss = []
xx_submit = []
N = 5
skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}


for k, (train_in, test_in) in enumerate(skf.split(train_csr, train_y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = train_csr[train_in], train_csr[test_in], train_y[train_in], train_y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    print(f1_score(y_test, np.where(gbm.predict(X_test, num_iteration=gbm.best_iteration)>0.4, 1,0)))
    xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
    xx_submit.append(gbm.predict(predict_csr, num_iteration=gbm.best_iteration))

print('train_logloss:', np.mean(xx_logloss))


s = 0
for i in xx_submit:
    s = s + i

test['label'] = list(s / N)
test['label'] = test['label'].apply(lambda x: round(x))
print('test_logloss:', np.mean(test.label))
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
test['label'].to_csv('baseline%s.csv' % now,index = False)