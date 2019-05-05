
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn import metrics
import warnings
import datetime
import utils
import time
import gc
from sklearn.feature_selection import chi2, SelectPercentile

import zipfile


# In[2]:


train_old = pd.read_csv('train_old.csv')
validate_old = pd.read_csv('valid_old.csv')
test_old = pd.read_csv('test_old.csv')

train_new = pd.read_csv('train_new.csv')
validate_new = pd.read_csv('valid_new.csv')
test_new = pd.read_csv('test_new.csv')


# In[6]:


def lgb_model(train_data, validate_data, test_data, parms, n_folds=5):
    
    columns = train_data.columns
    remove_columns = ["label",'prefix']
    features_columns = [column for column in columns if column not in remove_columns]
    
    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_data_length = validate_data.shape[0]
    validate_features = validate_data[features_columns]
    validate_labels = validate_data["label"]
    test_features = test_data[features_columns]
    test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True)

    clf = lgb.LGBMClassifier(**parms)
    kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    best_score = []
    loss = 0

    for train_index, test_index in kfold:
        k_x_train = train_features.loc[train_index]
        k_y_train = train_labels.loc[train_index]
        k_x_test = train_features.loc[test_index]
        k_y_test = train_labels.loc[test_index]

        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=100, verbose=True)

        best_score.append(lgb_clf.best_score_['valid']['binary_logloss'])
        loss += lgb_clf.best_score_['valid']['binary_logloss']
        print(best_score)
        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]

        preds_list.append(preds)

    print('logloss:', best_score, loss/5)
    
    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)
    
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    
    preds_df["mean"].to_csv('mean%s.csv' % now, index=False,header=False)
    
    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= 0.4 else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]


    f_score = f1_score(validate_labels, validate_preds["mean"])

    print(f1_score(k_y_test, np.where(lgb_clf.predict_proba(k_x_test, num_iteration=lgb_clf.best_iteration_)[:, 1]>0.4, 1,0)))

    # print('validata_logloss:', validate_preds["mean"])
    print("The validate data's f1_score is {}".format(f_score))

    # predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})
    # print('test_mean:', np.mean(predictions))
    
    # now = datetime.datetime.now()
    # now = now.strftime('%m-%d-%H-%M')

    # predictions.to_csv("predict%s.csv" % now, index=False, header=False)
        
    lgb_predictors = [i for i in train_data[features_columns].columns]
    lgb_feat_imp = pd.Series(lgb_clf.feature_importances_,lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('lgb_feat_imp_3%s.csv'% now)
    # lgb_predictors = [i for i in train_data[features_columns].columns]
    # #lgb_feat_imp = pd.Series(lgb_clf.feature_importances_,lgb_predictors).sort_values(ascending=False)
    # #lgb_feat_imp.to_csv('lgb_feat_imp_3.csv')
    # a = zipfile.ZipFile('test%s.zip' % now,'w')
    # a.write("predict%s.csv" % now)
    # a.close()
    
    
def model_main_old():
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False
    }

    lgb_model(train_old, validate_old, test_old, lgb_parms)

def model_main_new():
    lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2018,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False
    }    
    lgb_model(train_new, validate_new, test_new, lgb_parms)


# In[ ]:


import time
if __name__ == "__main__":
    t0 = time.time()
    model_main_old()
    print(time.time() - t0)


# In[4]:


import time
if __name__ == "__main__":
    t0 = time.time()
    model_main_new()
    print(time.time() - t0)


# In[3]:


mean_old = pd.read_csv('mean11-18-14-30.csv',header=None)
# 3万验证 12万测试


# In[11]:


mean_old = pd.read_csv('mean11-19-22-43.csv',header=None)


# In[ ]:





# In[4]:


mean_old_vali = mean_old[:30000]


# In[9]:


mean_old_vali.mean()


# In[7]:


mean_old_test = mean_old[30000:]


# In[8]:


mean_old_test.mean()


# In[10]:


mean_old_vali.columns=['mean']


# In[11]:


mean_new = pd.read_csv('mean11-18-21-47.csv',header=None)


# In[12]:


mean_new_vali = mean_new[:20000]


# In[13]:


mean_new_test = mean_new[20000:]


# In[14]:


mean_new_vali.columns=['mean']


# In[17]:


vali = utils.read_txt('../data_vali.txt')
train = utils.read_txt('../data_train.txt')
test = utils.read_txt('../data_test.txt',is_label=False)


# In[18]:


train_y = train['label'].astype(int).values
valid_y = vali['label'].astype(int).values


# In[22]:


valid_y.shape


# In[23]:


vali_30000_y = valid_y[:30000]
vali_20000_y = valid_y[30000:]


# In[18]:


mean_vali = pd.concat([mean_old_vali,mean_new_vali],axis=0,ignore_index=True)


# In[19]:


mean_vali.columns=['mean']


# In[20]:


mean_vali.mean()


# In[21]:


pred = (mean_vali['mean']>0.4).astype(np.uint32)


# In[23]:


pred.mean()


# In[24]:


f1_score(valid_y,pred)


# In[30]:


#分别判断阈值
for thr in np.arange(0.3,0.7,0.01):
    pred = (mean_old_vali['mean']>thr).astype(np.uint32)
    print(float("%0.4f"%thr),":valid f1_score: ",f1_score(vali_30000_y, pred),np.mean(pred))

for thr in np.arange(0.3,0.7,0.01):
    pred = (mean_new_vali['mean']>thr).astype(np.uint32)
    print(float("%0.4f"%thr),":valid f1_score: ",f1_score(vali_20000_y,pred),np.mean(pred))

#最后的最佳阈值还是0.38左右
#重复数据均值0.4139
#新数据均值0.43165


# In[34]:


mean_old_test.shape


# In[36]:


mean_new_test.shape


# In[25]:


mean_test = pd.concat([mean_old_test,mean_new_test],axis=0,ignore_index=True)


# In[40]:


mean_test


# In[26]:


mean_test.columns=['mean']


# In[27]:


mean_test.mean()


# In[50]:


mean_test['mean'] = mean_test['mean'].apply(lambda item: 1 if item >= 0.38 else 0)


# In[52]:


#当阈值为0.38
mean_test.mean()


# In[53]:


mean_test.to_csv('d_lgb_11182207.csv',index=False,header=False)


# In[55]:


a = zipfile.ZipFile('d_lgb.zip','w')
a.write("d_lgb_11182207.csv")
a.close()


# In[56]:


predict_pq = pd.read_csv('predict11-18-17-39.csv',header=None)


# In[58]:


predict_pq.head(10)


# In[59]:


mean_test.head(10)


# In[31]:


mean_test = pd.concat([mean_old_test,mean_new_test],axis=0,ignore_index=True)
mean_test.columns=['mean']
mean_test['mean'] = mean_test['mean'].apply(lambda item: 1 if item >= 0.4 else 0)


# In[29]:


pred = (mean_test['mean']>0.4).astype(np.uint32)


# In[30]:


pred.mean()


# In[32]:


#当阈值为0.4时
mean_test.mean()


# In[65]:


mean_test.to_csv('d_lgb_11182207_threshold0.4.csv',index=False,header=False)


# In[33]:


mean_test.head()


# In[ ]:


a = zipfile.ZipFile('d_lgb_0.4.zip','w')
a.write("d_lgb_11182207_threshold0.4.csv")
a.close()


# In[12]:


predict = pd.read_csv('d_lgb_11182207_threshold0.4.csv',header=None)


# In[13]:


predict.mean()


# In[ ]:





# In[2]:


mean_old = pd.read_csv('mean11-19-22-43.csv',header=None)


# In[3]:


mean_old_vali = mean_old[:30000]
mean_old_test = mean_old[30000:]


# In[4]:


mean_new = pd.read_csv('mean11-19-22-49.csv',header=None)


# In[5]:


mean_new_vali = mean_new[:20000]
mean_new_test = mean_new[20000:]


# In[6]:


mean_old_vali.columns=['mean']
mean_new_vali.columns=['mean']
#分别判断阈值
for thr in np.arange(0.3,0.5,0.01):
    pred1 = (mean_old_vali['mean']>thr).astype(np.uint32)
    print(float("%0.4f"%thr),":valid f1_score: ",f1_score(vali_30000_y, pred1),np.mean(pred))

for thr in np.arange(0.3,0.5,0.01):
    pred2 = (mean_new_vali['mean']>thr).astype(np.uint32)
    print(float("%0.4f"%thr),":valid f1_score: ",f1_score(vali_20000_y,pred2),np.mean(pred))


# In[7]:


pred1 = (mean_old_vali['mean']>0.34).astype(np.uint32)


# In[8]:


pred2 = (mean_new_vali['mean']>0.38).astype(np.uint32)


# In[9]:


vali_pred = pd.concat([pred1,pred2],axis=0,ignore_index=True)


# In[37]:


f1_score(valid_y,vali_pred)


# In[10]:


mean_old_test.columns=['mean']
mean_new_test.columns=['mean']
pred1 = (mean_old_test['mean']>0.4).astype(np.uint32)
pred2 = (mean_new_test['mean']>0.4).astype(np.uint32)
test_pred = pd.concat([pred1,pred2],axis=0,ignore_index=True)


# In[11]:


test_pred.mean()


# In[ ]:




