# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 15:51:16 2018

@author: PQ
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('validate.csv')


train = train.fillna(0)
val = val.fillna(0)
test = test.fillna(0)


train_data = pd.concat([train, val], axis=0, ignore_index=True)
test_data_= test


X = np.array(train_data.drop(['label'], axis = 1))
y = np.array(train_data['label'])
X_test_ = np.array(test_data_)
print('================================')
print(X.shape)
print(y.shape)
print('================================')


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

for k, (train_in, test_in) in enumerate(skf.split(X, y)):
    print('train _K_ flod', k)
    X_train, X_test, y_train, y_test = X[train_in], X[test_in], y[train_in], y[test_in]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    print(f1_score(y_test, np.where(gbm.predict(X_test, num_iteration=gbm.best_iteration)>0.5, 1,0)))
    xx_logloss.append(gbm.best_score['valid_0']['binary_logloss'])
    xx_submit.append(gbm.predict(X_test_, num_iteration=gbm.best_iteration))

print('train_logloss:', np.mean(xx_logloss))
s = 0
for i in xx_submit:
    s = s + i

test_data_['label'] = list(s / N)
test_data_['label'] = test_data_['label'].apply(lambda x: round(x))
print('test_logloss:', np.mean(test_data_.label))
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
test_data_['label'].to_csv('result%s.csv'% now,index = False)