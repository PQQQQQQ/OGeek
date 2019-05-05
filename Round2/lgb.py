
#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn import metrics
import warnings
import datetime
import utils
import time
import gc
from featexp import get_trend_stats
import zipfile

warnings.filterwarnings('ignore')


# In[52]:


ctr_df = pd.read_csv('ctr_df_x.csv')
ctr_df = ctr_df.drop(columns=['cate_prefix', 'cate_title',
       'prefix_num', 'title_num', 'query_prediction', 'prefix_click',
       'prefix_count', 'prefix_ctr', 'title_click', 'title_count',
       'title_ctr', 'tag_click', 'tag_count', 'tag_ctr',
       'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr',
       'prefix_tag_click', 'prefix_tag_count', 'prefix_tag_ctr',
       'title_tag_click', 'title_tag_count', 'title_tag_ctr',
       'cate_prefix_click', 'cate_prefix_count', 'cate_prefix_ctr',
       'cate_title_click', 'cate_title_count', 'cate_title_ctr',
       'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
       'cate_prefix_cate_title_ctr', 'cate_prefix_tag_click',
       'cate_prefix_tag_count', 'cate_prefix_tag_ctr',
       'cate_title_tag_click', 'cate_title_tag_count',
       'cate_title_tag_ctr', 'prefix_title_tag_click',
       'prefix_title_tag_count', 'prefix_title_tag_ctr',
       'prefix_num_title_num_tag_click', 'prefix_num_title_num_tag_count',
       'prefix_num_title_num_tag_ctr'])


# In[53]:


prefix_df = pd.read_csv('prefix_df_x.csv')
ctr_df = pd.read_csv('ctr_df_x.csv')
#cvr_df = pd.read_csv('../cvr_df.csv')
query_df = pd.read_csv('query_df_x.csv')
text_df = pd.read_csv('text_df_x.csv')
stat_df = pd.read_csv('stat_df_x.csv')

com_prefix_df = pd.read_csv('comp_prefix_df_x.csv')


# In[54]:


nunique_df = pd.read_csv('nunique_df_x.csv')
stat_df['prob_max_ratio'] = stat_df['prob_max'] / (stat_df['prob_sum']+0.000001)


# In[79]:


prefix_df.columns.values


# In[80]:


query_df.columns.values


# In[81]:


com_prefix_df.columns.values


# In[68]:


text_df.columns.values


# In[82]:


nunique_df.columns.values


# In[66]:


stat_df.columns.values


# In[16]:


nunique_df.columns.values


# In[6]:


# ctr_df['query_prediction'].isnull().sum()


# In[65]:


# ctr_df.isnull().sum()


# In[55]:


drop_columns = ['prefix', 'title']
prefix_df = prefix_df.drop(columns=drop_columns)
drop_columns_p = ['complete_prefix', 'title']
com_prefix_df = com_prefix_df.drop(columns=drop_columns_p)
drop_columns_1 = ['prefix', 'query_prediction', 'tag', 'title', 'label']
text_df = text_df.drop(columns=drop_columns_1)
# 'prob_sum', 'prob_mean'
drop_columns_s = ['label', 'tag', 'complete_prefix', 'prefix_word_num', 'title_len', 'query_length','prob_mean', 'prob_std','prob_sum','prob_min']
stat_df = stat_df.drop(columns=drop_columns_s)
#nunique_df = nunique_df.drop(columns=['prefix', 'query_prediction', 'title', 'tag', 'label'])


# In[40]:


# query_df.isnull().sum()


# In[56]:


df = pd.concat([ctr_df, prefix_df, query_df, text_df, stat_df, nunique_df , com_prefix_df], axis=1)
del ctr_df, prefix_df, query_df, text_df, stat_df, nunique_df , com_prefix_df
gc.collect()


# In[6]:


#df[df.cate_prefix == 9]


# In[1]:


# df[df.prefix == '王者荣耀']


# In[87]:


df.columns.values


# In[7]:


df.shape


# In[5]:


# df = df.drop(columns=['complete_prefix.1'])


# In[62]:


#'query_prediction' 'prefix_num', 'title_num', 
drop_columns_2 = ['prefix',  'title','cut_prefix', 'cut_title', 'cut_query_prediction', 'words']
df = df.drop(columns=drop_columns_2)


# In[63]:


df.columns.values


# In[64]:


df.shape


# In[7]:


# df = df.drop(columns=['complete_prefix_title','complete_prefix','complete_prefix_tag','prob_min','prob_std'])


# In[65]:


df['tag']=le.fit_transform(df['tag'])


# In[66]:


for col in df.columns:
    print(col,':',df[col].isnull().sum())
for col in df.columns:
    df[col]=df[col].fillna(0)#df[col].mean()
for col in df.columns:
    print(col,':',df[col].isnull().sum())   


# In[22]:


df.shape


# In[99]:


# df  = df.drop(columns=['prob_max_ratio'])


# In[67]:


train_df_length = 2000000
validate_df_length = 50000


train_data = df[:train_df_length]
train_data["label"] = train_data["label"].apply(int)

validate_data = df[train_df_length:train_df_length + validate_df_length]
validate_data["label"] = validate_data["label"].apply(int)

test_data = df[train_df_length + validate_df_length:]
test_data = test_data.drop(columns=["label"])


# In[12]:


df.shape


# In[ ]:





# In[13]:


# train_data.shape


# In[68]:


def my_f1_score(labels, preds):
    preds=[1 if i>=0.4 else 0 for i in preds]
    return 'f1_score', f1_score(labels,preds), True
def lgb_model_2(train_data, validate_data, test_data, parms, n_folds=2):
    categorical_feature=['tag']# 
    columns = train_data.columns
    print('train shape:',train_data.shape)
    print('train feature:',columns)
    start_time=time.time()
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_data_length = validate_data.shape[0]
    validate_features = validate_data[features_columns]
    validate_labels = validate_data["label"]
    validate_df=validate_features.values
    valid_label=validate_labels.values
    test_features = test_data[features_columns]
    test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True)

    clf = lgb.LGBMClassifier(**parms)
    # kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    # kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    best_score = []
    loss = 0
    k_x_train0, k_x_test0, k_y_train0, k_y_test0= train_test_split(train_features, train_labels, test_size=0.2, random_state=2018)

    k_x_train1, k_x_test1, k_y_train1, k_y_test1 = train_test_split(train_features, train_labels, test_size=0.2, random_state=2)
    
#     valid_in_cv = k_x_test0[k_x_test0.index >= 2000000]
#     valid_not_in_cv = validate_data[validate_data.index != valid_in_cv.index]
#     valid_not_in_cv_x = valid_not_in_cv[feature_columns]
#     valid_not_in_cv_y = valid_not_in_cv['label']
    #
    for (k_x_train, k_x_test, k_y_train, k_y_test) in [(k_x_train1, k_x_test1, k_y_train1, k_y_test1),(k_x_train0, k_x_test0, k_y_train0, k_y_test0)]:
        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=50,
                          #eval_metric=metrix_function,
#                           feature_name=feature_name,  
                          categorical_feature=categorical_feature,
                          verbose=10)
        best_score.append(lgb_clf.best_score_['valid']['binary_logloss'])
        loss += lgb_clf.best_score_['valid']['binary_logloss']
        print(best_score)
        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]
        valid_pred=lgb_clf.predict_proba(validate_df, num_iteration=lgb_clf.best_iteration_)[:,1]
        valid_pred=[1 if i>=0.4 else 0 for i in valid_pred]
        print('valid f_socre:',f1_score(valid_label,valid_pred))
        
        
        valid_not_in_cv_index = k_x_test[k_x_test.index >= 2000000]        
        valid_not_in_cv = validate_data[(validate_data.index).isin(valid_not_in_cv_index.index)]
        valid_not_in_cv_y = valid_not_in_cv['label'].values.astype(int)
        #valid_not_in_cv_y = k_y_test[k_y_test.index >= 2000000]
        valid_not_in_cv_pred = [valid_pred[x] for x in (np.array(valid_not_in_cv.index)-2000000)]
        print("not cv valid f_score:",f1_score(valid_not_in_cv_y,np.array(valid_not_in_cv_pred)))
        
        preds_list.append(preds)

    print('logloss:', best_score, loss/2)
    
    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)
    
    y_prob_test=preds_df["mean"][50000:]
    y_prob_val=preds_df["mean"][:50000]
    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= 0.4 else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    f_score = f1_score(validate_labels, validate_preds["mean"])

    # print('validata_logloss:', validate_preds["mean"])
    print("The validate data's f1_score is {}".format(f_score))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})
    print('test_mean:', np.mean(predictions))
    
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    
    y_prob_test.to_csv("prob_test%f-%s.csv" % (f_score,now),index=False, header=False)
    
    y_prob_val.to_csv("prob_valid%f-%s.csv" % (f_score,now),index=False, header=False)
    
    lgb_predictors = [i for i in train_data[features_columns].columns]
    lgb_feat_imp = pd.Series(lgb_clf.feature_importances_, lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('lgb_feat_imp_1.csv')
    
    predictions.to_csv("predict%f-%s.csv" % (f_score,now), index=False, header=False)
    zip_predict = zipfile.ZipFile('predict%f-%s.zip' % (f_score,now),'w')
    zip_predict.write("predict%f-%s.csv" % (f_score,now))
    zip_predict.close()
    end_time=time.time()
    print('time:',(end_time-start_time)/60)
    return y_prob_val,y_prob_test

lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves":127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 120,
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


# In[69]:


y_prob_valid,y_prob_test=lgb_model_2(train_data, validate_data, test_data, lgb_parms)


# for i in np.arange(0.35,0.5,0.001):
#     label_valid=np.array([1 if y>=i else 0 for y in y_prob_valid])
#     label_test=np.array([1 if y>=i else 0 for y in y_prob_test])
#     print(i,f1_score(validate_data["label"].values, label_valid),np.mean(label_valid),np.mean(label_test))


for thr in np.linspace(0.35,0.55,40):
    pred = (y_prob_valid>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(validate_data["label"].values, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[10]:


# columns = train_data.columns
# # print('train shape:',train_data.shape)
# # print('train feature:',columns)
# start_time=time.time()
# remove_columns = ["label"]
# features_columns = [column for column in columns if column not in remove_columns]

# train_data_1 = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
# train_features = train_data_1[features_columns]
# train_labels = train_data_1["label"]
#k_x_train2, k_x_test2, k_y_train2, k_y_test2= train_test_split(train_features, train_labels, test_size=0.2, random_state=2018)


# valid_not_in_cv_index = k_x_test2[k_x_test2.index >= 2000000]        
# valid_not_in_cv = validate_data[(validate_data.index).isin(valid_not_in_cv_index.index)]
# valid_not_in_cv_y = valid_not_in_cv['label'].values.astype(int)
# y_pred_valid = (y_prob_valid>thr).astype(np.uint32)
# valid_not_in_cv_pred = [y_pred_valid[x] for x in (np.array(valid_not_in_cv.index)-2000000)]
# print("not cv valid f_score:",f1_score(valid_not_in_cv_y,np.array(valid_not_in_cv_pred)))


# # 特征筛选

# In[12]:


feature = ['tag', 'cate_prefix', 'cate_title', 'prefix_click', 'prefix_count',
       'prefix_ctr', 'title_click', 'title_count', 'title_ctr',
       'tag_click', 'tag_count', 'tag_ctr', 'prefix_title_click',
       'prefix_title_count', 'prefix_title_ctr', 'prefix_tag_click',
       'prefix_tag_count', 'prefix_tag_ctr', 'title_tag_click',
       'title_tag_count', 'title_tag_ctr', 'cate_prefix_click',
       'cate_prefix_count', 'cate_prefix_ctr', 'cate_title_click',
       'cate_title_count', 'cate_title_ctr',
       'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
       'cate_prefix_cate_title_ctr', 'cate_prefix_tag_click',
       'cate_prefix_tag_count', 'cate_prefix_tag_ctr',
       'cate_title_tag_click', 'cate_title_tag_count',
       'cate_title_tag_ctr', 'prefix_title_tag_click',
       'prefix_title_tag_count', 'prefix_title_tag_ctr',
       'prefix_num_title_num_tag_click', 'prefix_num_title_num_tag_count',
       'prefix_num_title_num_tag_ctr', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'max_similar', 'mean_similar',
       'weight_similar', 'title_pred_score', 'title_unseen_nword',
       'pred_freq_std', 'pred_freq_mean', 'nword_title_unseen_in_prefix',
       'pred_freq_sum', 'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'max_query_ratio', 'title_word_num', 'small_query_num', 'prob_max',
       'prob_max_ratio', 'complete_prefix_click', 'complete_prefix_count',
       'complete_prefix_ctr', 'complete_prefix_title_click',
       'complete_prefix_title_count', 'complete_prefix_title_ctr',
       'complete_prefix_tag_click', 'complete_prefix_tag_count',
       'complete_prefix_tag_ctr', 'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c']


# In[13]:


columns = train_data.columns

remove_columns = ["label"]
features_columns = [column for column in columns if column not in remove_columns]

#train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
train_features = train_data[features_columns]
train_labels = train_data["label"]

#validate_data_length = validate_data.shape[0]
validate_features = validate_data[features_columns]
validate_labels = validate_data["label"]


# In[14]:


x = pd.concat([train_features,train_labels],axis=1)
y = pd.concat([validate_features,validate_labels],axis=1)


# In[30]:


from featexp import get_trend_stats
from featexp import get_univariate_plots
get_univariate_plots(data = x, target_col='label',                      data_test = y, features_list=['prob_max'],bins=10)


# In[31]:


stats = get_trend_stats(data=x, target_col='label',                      data_test=y)


# In[32]:


stats[stats.Trend_correlation < 0.6]


# In[13]:


df.shape


# In[14]:


df = df.drop(columns=['title_tag_count'])


# In[15]:


train_df_length = 2000000
validate_df_length = 50000


train_data = df[:train_df_length]
train_data["label"] = train_data["label"].apply(int)

validate_data = df[train_df_length:train_df_length + validate_df_length]
validate_data["label"] = validate_data["label"].apply(int)

test_data = df[train_df_length + validate_df_length:]
test_data = test_data.drop(columns=["label"])


# In[16]:


train_data.shape


# In[17]:


y_prob_valid,y_prob_test=lgb_model_2(train_data, validate_data, test_data, lgb_parms)


# for i in np.arange(0.35,0.5,0.001):
#     label_valid=np.array([1 if y>=i else 0 for y in y_prob_valid])
#     label_test=np.array([1 if y>=i else 0 for y in y_prob_test])
#     print(i,f1_score(validate_data["label"].values, label_valid),np.mean(label_valid),np.mean(label_test))


for thr in np.linspace(0.35,0.55,40):
    pred = (y_prob_valid>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(validate_data["label"].values, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # 不要kmeans

# In[41]:


df.columns.values


# In[42]:


df.shape


# In[43]:


df = df.drop(columns=['cate_prefix', 'cate_title','cate_prefix_click',
       'cate_prefix_count', 'cate_prefix_ctr', 'cate_title_click',
       'cate_title_count', 'cate_title_ctr',
       'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
       'cate_prefix_cate_title_ctr', 'cate_prefix_tag_click',
       'cate_prefix_tag_count', 'cate_prefix_tag_ctr',
       'cate_title_tag_click', 'cate_title_tag_count',
       'cate_title_tag_ctr'])


# In[44]:


train_df_length = 2000000
validate_df_length = 50000


train_data = df[:train_df_length]
train_data["label"] = train_data["label"].apply(int)

validate_data = df[train_df_length:train_df_length + validate_df_length]
validate_data["label"] = validate_data["label"].apply(int)

test_data = df[train_df_length + validate_df_length:]
test_data = test_data.drop(columns=["label"])


# In[45]:


train_data.shape


# In[46]:


def my_f1_score(labels, preds):
    preds=[1 if i>=0.4 else 0 for i in preds]
    return 'f1_score', f1_score(labels,preds), True
def lgb_model_2(train_data, validate_data, test_data, parms, n_folds=2):
    categorical_feature=['tag']# 
    columns = train_data.columns
    print('train shape:',train_data.shape)
    print('train feature:',columns)
    start_time=time.time()
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_data_length = validate_data.shape[0]
    validate_features = validate_data[features_columns]
    validate_labels = validate_data["label"]
    validate_df=validate_features.values
    valid_label=validate_labels.values
    test_features = test_data[features_columns]
    test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True)

    clf = lgb.LGBMClassifier(**parms)
    # kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    # kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    best_score = []
    loss = 0
    k_x_train0, k_x_test0, k_y_train0, k_y_test0= train_test_split(train_features, train_labels, test_size=0.2, random_state=2018)

    k_x_train1, k_x_test1, k_y_train1, k_y_test1 = train_test_split(train_features, train_labels, test_size=0.2, random_state=2)
    
#     valid_in_cv = k_x_test0[k_x_test0.index >= 2000000]
#     valid_not_in_cv = validate_data[validate_data.index != valid_in_cv.index]
#     valid_not_in_cv_x = valid_not_in_cv[feature_columns]
#     valid_not_in_cv_y = valid_not_in_cv['label']
    #
    for (k_x_train, k_x_test, k_y_train, k_y_test) in [(k_x_train1, k_x_test1, k_y_train1, k_y_test1),(k_x_train0, k_x_test0, k_y_train0, k_y_test0)]:
        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=50,
                          #eval_metric=metrix_function,
#                           feature_name=feature_name,  
                          categorical_feature=categorical_feature,
                          verbose=10)
        best_score.append(lgb_clf.best_score_['valid']['binary_logloss'])
        loss += lgb_clf.best_score_['valid']['binary_logloss']
        print(best_score)
        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]
        valid_pred=lgb_clf.predict_proba(validate_df, num_iteration=lgb_clf.best_iteration_)[:,1]
        valid_pred=[1 if i>=0.4 else 0 for i in valid_pred]
        print('valid f_socre:',f1_score(valid_label,valid_pred))
        
        
        valid_not_in_cv_index = k_x_test[k_x_test.index >= 2000000]        
        valid_not_in_cv = validate_data[(validate_data.index).isin(valid_not_in_cv_index.index)]
        valid_not_in_cv_y = valid_not_in_cv['label'].values.astype(int)
        #valid_not_in_cv_y = k_y_test[k_y_test.index >= 2000000]
        valid_not_in_cv_pred = [valid_pred[x] for x in (np.array(valid_not_in_cv.index)-2000000)]
        print("not cv valid f_score:",f1_score(valid_not_in_cv_y,np.array(valid_not_in_cv_pred)))
        
        preds_list.append(preds)

    print('logloss:', best_score, loss/2)
    
    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)
    
    y_prob_test=preds_df["mean"][50000:]
    y_prob_val=preds_df["mean"][:50000]
    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= 0.4 else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    f_score = f1_score(validate_labels, validate_preds["mean"])

    # print('validata_logloss:', validate_preds["mean"])
    print("The validate data's f1_score is {}".format(f_score))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})
    print('test_mean:', np.mean(predictions))
    
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    
    y_prob_test.to_csv("prob_test%f-%s.csv" % (f_score,now),index=False, header=False)
    
    y_prob_val.to_csv("prob_valid%f-%s.csv" % (f_score,now),index=False, header=False)
    
    lgb_predictors = [i for i in train_data[features_columns].columns]
    lgb_feat_imp = pd.Series(lgb_clf.feature_importances_, lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('lgb_feat_imp_1.csv')
    
    predictions.to_csv("predict%f-%s.csv" % (f_score,now), index=False, header=False)
    zip_predict = zipfile.ZipFile('predict%f-%s.zip' % (f_score,now),'w')
    zip_predict.write("predict%f-%s.csv" % (f_score,now))
    zip_predict.close()
    end_time=time.time()
    print('time:',(end_time-start_time)/60)
    return y_prob_val,y_prob_test

lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves":127,
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


# In[47]:


y_prob_valid,y_prob_test=lgb_model_2(train_data, validate_data, test_data, lgb_parms)


# for i in np.arange(0.35,0.5,0.001):
#     label_valid=np.array([1 if y>=i else 0 for y in y_prob_valid])
#     label_test=np.array([1 if y>=i else 0 for y in y_prob_test])
#     print(i,f1_score(validate_data["label"].values, label_valid),np.mean(label_valid),np.mean(label_test))


for thr in np.linspace(0.35,0.55,40):
    pred = (y_prob_valid>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(validate_data["label"].values, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[48]:


cvr_df = pd.read_csv('../cvr_df.csv')


# In[49]:


cvr_df.columns.values


# In[97]:


predict = pd.read_csv('predict0.780627-11-27-15-11.csv',header=None)


# In[98]:


predict.mean()


# # 增量训练

# In[ ]:


def my_f1_score(labels, preds):
    preds=[1 if i>=0.4 else 0 for i in preds]
    return 'f1_score', f1_score(labels,preds), True
def lgb_model_2(train_data, validate_data, test_data, parms, n_folds=2):
    categorical_feature=['tag','cate_title','cate_prefix']# 
    columns = train_data.columns
    print('train shape:',train_data.shape)
    print('train feature:',columns)
    start_time=time.time()
    remove_columns = ["label"]
    features_columns = [column for column in columns if column not in remove_columns]

    train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
    train_features = train_data[features_columns]
    train_labels = train_data["label"]

    validate_data_length = validate_data.shape[0]
    validate_features = validate_data[features_columns]
    validate_labels = validate_data["label"]
    validate_df=validate_features.values
    valid_label=validate_labels.values
    test_features = test_data[features_columns]
    test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True)

    clf = lgb.LGBMClassifier(**parms)
    # kfolder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2018)
    # kfold = kfolder.split(train_features, train_labels)

    preds_list = list()
    best_score = []
    loss = 0
    k_x_train0, k_x_test0, k_y_train0, k_y_test0= train_test_split(train_features, train_labels, test_size=0.2, random_state=2018)

    k_x_train1, k_x_test1, k_y_train1, k_y_test1 = train_test_split(train_features, train_labels, test_size=0.2, random_state=2)
    #
    for (k_x_train, k_x_test, k_y_train, k_y_test) in [(k_x_train1, k_x_test1, k_y_train1, k_y_test1),(k_x_train0, k_x_test0, k_y_train0, k_y_test0)]:
        lgb_clf = clf.fit(k_x_train, k_y_train,
                          eval_names=["train", "valid"],
                          eval_metric="logloss",
                          eval_set=[(k_x_train, k_y_train),
                                    (k_x_test, k_y_test)],
                          early_stopping_rounds=50,
                          #eval_metric=metrix_function,
#                           feature_name=feature_name,  
                          categorical_feature=categorical_feature,
                          verbose=10)
        best_score.append(lgb_clf.best_score_['valid']['binary_logloss'])
        loss += lgb_clf.best_score_['valid']['binary_logloss']
        print(best_score)
        preds = lgb_clf.predict_proba(test_features, num_iteration=lgb_clf.best_iteration_)[:, 1]
        valid_pred=lgb_clf.predict_proba(validate_df, num_iteration=lgb_clf.best_iteration_)[:,1]
        valid_pred=[1 if i>=0.4 else 0 for i in valid_pred]
        print('valid f_socre:',f1_score(valid_label,valid_pred))
        
        preds_list.append(preds)

    print('logloss:', best_score, loss/2)
    
    length = len(preds_list)
    preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

    preds_df = pd.DataFrame(data=preds_list)
    preds_df = preds_df.T
    preds_df.columns = preds_columns
    preds_df = preds_df.copy()
    preds_df["mean"] = preds_df.mean(axis=1)
    
    y_prob_test=preds_df["mean"][50000:]
    y_prob_val=preds_df["mean"][:50000]
    preds_df["mean"] = preds_df["mean"].apply(lambda item: 1 if item >= 0.4 else 0)

    validate_preds = preds_df[:validate_data_length]
    test_preds = preds_df[validate_data_length:]

    f_score = f1_score(validate_labels, validate_preds["mean"])

    # print('validata_logloss:', validate_preds["mean"])
    print("The validate data's f1_score is {}".format(f_score))

    predictions = pd.DataFrame({"predicted_score": test_preds["mean"]})
    print('test_mean:', np.mean(predictions))
    
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    
    y_prob_test.to_csv("prob_test%f-%s.csv" % (f_score,now),index=False, header=False)
    
    y_prob_val.to_csv("prob_valid%f-%s.csv" % (f_score,now),index=False, header=False)
    
    lgb_predictors = [i for i in train_data[features_columns].columns]
    lgb_feat_imp = pd.Series(lgb_clf.feature_importances_, lgb_predictors).sort_values(ascending=False)
    lgb_feat_imp.to_csv('lgb_feat_imp_1.csv')
    
    predictions.to_csv("predict%f-%s.csv" % (f_score,now), index=False, header=False)
    zip_predict = zipfile.ZipFile('predict%f-%s.zip' % (f_score,now),'w')
    zip_predict.write("predict%f-%s.csv" % (f_score,now))
    zip_predict.close()
    end_time=time.time()
    print('time:',(end_time-start_time)/60)
    return y_prob_val,y_prob_test

lgb_parms = {
        "boosting_type": "gbdt",
        "num_leaves":127,
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

y_prob_valid,y_prob_test=lgb_model_2(train_data, validate_data, test_data, lgb_parms)


# for i in np.arange(0.35,0.5,0.001):
#     label_valid=np.array([1 if y>=i else 0 for y in y_prob_valid])
#     label_test=np.array([1 if y>=i else 0 for y in y_prob_test])
#     print(i,f1_score(validate_data["label"].values, label_valid),np.mean(label_valid),np.mean(label_test))


for thr in np.linspace(0.35,0.55,40):
    pred = (y_prob_valid>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(validate_data["label"].values, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))

