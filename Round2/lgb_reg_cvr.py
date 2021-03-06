
#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import utils
import numpy as np
import pandas as pd
import jieba
import string
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import f1_score,accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import gc
import os

import scipy.special as special
import math
from math import log


# In[3]:


import unicodedata
from itertools import groupby
import re

def text_clean(text,patterns=[]):
    patterns = [('','')]
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    """ Simple text clean up process"""
    clean = text.lower()
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return clean
def my_f1_score(labels, preds):
    return 'f1_score', f1_score(labels,preds.round()), True


# In[4]:


# root_dir = '../../DataSets/oppo_data_ronud2_20181107/'

df_valid = utils.read_txt('../data_vali.txt')
df_train = utils.read_txt('../data_train.txt')
df_testa = utils.read_txt('../data_test.txt',is_label=False)

df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)
train_y = df_train['label'].astype(int).values
valid_y = df_valid['label'].astype(int).values

df_testa['label']=-1
df_train['dataset_type']=-1
df_valid['dataset_type']=-2
df_testa['dataset_type']=-3
df_data=pd.concat([df_train,df_valid,df_testa]).reset_index(drop=True)


# In[5]:


df_data["prefix"] = df_data["prefix"].apply(text_clean)
df_data["title"] = df_data["title"].apply(text_clean)


# In[6]:


def func(x):
    preds = x['query_prediction'].unique()
    preds = [_x for _x in preds if len(_x)>0]
    if len(preds)==0:
        return ''
    else:
        return preds[0]

df_tmp = df_data.groupby(['prefix']).apply(lambda x:func(x)).reset_index(drop=False).rename(columns={0:'cleaned_query_prediction'})
df_data = pd.merge(df_data,df_tmp,on='prefix',how='left')

df_data = df_data.drop(columns=['query_prediction'])
df_data = df_data.rename(columns={'cleaned_query_prediction':'query_prediction'})
df_data['query_prediction'] = df_data['query_prediction'].apply(lambda x: x if x!='' else '{}')


# In[10]:


# df_data[df_data.prefix == 'Dior']


# In[7]:


query_df = pd.read_csv('createfeature/query_df_x.csv')
prefix_df = pd.read_csv('createfeature/prefix_df_x.csv')
text_df = pd.read_csv('createfeature/text_df_x.csv')
ctr_df = pd.read_csv('createfeature/ctr_df_x.csv')
complete_prefix_df = pd.read_csv('createfeature/comp_prefix_df_x.csv')
#nunique_df = pd.read_csv('createfeature/nunique_df_x.csv')
stat_df = pd.read_csv('createfeature/stat_df_x.csv')


# In[8]:


drop_columns = ['prefix', 'title']
prefix_df = prefix_df.drop(columns=drop_columns)
drop_columns_p = ['title']
complete_prefix_df = complete_prefix_df.drop(columns=drop_columns_p)
drop_columns_1 = ['prefix', 'query_prediction', 'tag', 'title', 'label']
text_df = text_df.drop(columns=drop_columns_1)
# 'prob_sum', 'prob_mean'
drop_columns_s = ['label', 'tag', 'complete_prefix', 'prefix_word_num', 'title_len', 'query_length','prob_mean', 'prob_std','prob_sum','prob_min']
stat_df = stat_df.drop(columns=drop_columns_s)


# In[4]:


#ctr 保留'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
# ctr_df = ctr_df.drop(columns=['label', 'prefix', 'query_prediction', 'tag', 'title',
#        'prefix_click', 'prefix_count', 'prefix_ctr', 'title_click',
#        'title_count', 'title_ctr', 'tag_click', 'tag_count', 'tag_ctr',
#        'prefix_title_click', 'prefix_title_count', 'prefix_title_ctr',
#        'prefix_tag_click', 'prefix_tag_count', 'prefix_tag_ctr',
#        'title_tag_click', 'title_tag_count', 'title_tag_ctr',
#        'cate_prefix_click', 'cate_prefix_count', 'cate_prefix_ctr',
#        'cate_title_click', 'cate_title_count', 'cate_title_ctr',
#        'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
#        'cate_prefix_cate_title_ctr', 'cate_prefix_tag_click',
#        'cate_prefix_tag_count', 'cate_prefix_tag_ctr',
#        'cate_title_tag_click', 'cate_title_tag_count',
#        'cate_title_tag_ctr', 'prefix_title_tag_click',
#        'prefix_title_tag_count', 'prefix_title_tag_ctr',
#        'prefix_num_title_num_tag_click', 'prefix_num_title_num_tag_count',
#        'prefix_num_title_num_tag_ctr'])


# In[9]:


df_data['prefix_num'] = df_data['prefix'].apply(lambda x:len(x))
df_data['title_num'] = df_data['title'].apply(lambda x:len(x))


# In[10]:


df_data = pd.concat([df_data,query_df,prefix_df,text_df,complete_prefix_df,stat_df],axis=1)

del query_df,prefix_df,text_df,complete_prefix_df,stat_df
gc.collect()


# In[18]:


df_data.columns.values


# In[11]:


def clean_csr(csr_trn, csr_sub, min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    sub_min = {x for x in np.where(csr_sub.getnnz(axis=0) >= min_df)[0]}
    mask= [x for x in trn_min if x in sub_min]
    return csr_trn[:, mask], csr_sub[:, mask]
def statis_feat_smooth(df,feature,label = 'label'):
    df = df.groupby(feature)[label].agg(['sum','count']).reset_index()
    new_feat_name = feature + '_cvstat_cvr'
    I = df['count']
    C = df['sum']
    df.loc[:,new_feat_name] = C/(I+0.01)
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
    df_stas = df[[feature,new_feat_name]]
    return df_stas
def statis_feat_smooth_and_merge(df,df_val, feature,label = 'label'):
    df_stas = statis_feat_smooth(df,feature,label =label)
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)
    return df_val
    
class CVRCVStatistic2(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,min_count=0,kfold=10,is_test_use_val=True):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for concat_feat in self.feat_list:
            df_cv = df[['_index',concat_feat, self.label,'dataset_type']]
            new_feat_name = concat_feat + '_cvstat_cvr'
            self.cvr_feat_list.append(new_feat_name)
            if min_count>0:
                df_cv[concat_feat] = self._remove_lowcase(df_cv[concat_feat],min_count)
            # train val test
            training = df_cv[df_cv['dataset_type']==-1]  
            training = training.reset_index(drop=True)
            valid = df_cv[df_cv.dataset_type==-2]  
            trainval = df_cv[df_cv['dataset_type']!=-3]
            trainval = trainval.reset_index(drop=True)
            predict = df_cv[df_cv.dataset_type==-3]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(kfold,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = statis_feat_smooth_and_merge(X_train,X_val,concat_feat,label=self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)
            X_valid = statis_feat_smooth_and_merge(training, valid, concat_feat,label=self.label)
            if is_test_use_val:
                X_pred = statis_feat_smooth_and_merge(trainval, predict, concat_feat,label=self.label)
            else:
                X_pred = statis_feat_smooth_and_merge(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_valid,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del df_stas_feat['dataset_type']
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df
                
    def _remove_lowcase(self, se,min_count=5):
        count = dict(se.value_counts())
        se = se.map(lambda x : -1 if count[x]<min_count else x)
        return se
    
class CrossFeatCVRCVStatistic2(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,min_count=0,kfold=10,is_test_use_val=True):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2 in self.feat_list:
            concat_feat = feat1+'_'+feat2
            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)
            df_cv = df[['_index',concat_feat, self.label,'dataset_type']]
            new_feat_name = concat_feat + '_cvstat_cvr'
            self.cvr_feat_list.append(new_feat_name)
            if min_count>0:
                df_cv[concat_feat] = self._remove_lowcase(df_cv[concat_feat],min_count)
             # train val test
            training = df_cv[df_cv['dataset_type']==-1]  
            training = training.reset_index(drop=True)
            valid = df_cv[df_cv.dataset_type==-2]  
            trainval = df_cv[df_cv['dataset_type']!=-3]
            trainval = trainval.reset_index(drop=True)
            predict = df_cv[df_cv.dataset_type==-3]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(kfold,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = statis_feat_smooth_and_merge(X_train,X_val,concat_feat,label=self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)
            
            X_valid = statis_feat_smooth_and_merge(training, valid, concat_feat,label=self.label)
            if is_test_use_val:
                X_pred = statis_feat_smooth_and_merge(trainval, predict, concat_feat,label=self.label)
            else:
                X_pred = statis_feat_smooth_and_merge(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_valid,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del df_stas_feat['dataset_type']
            del training
            del predict
            del df[concat_feat]
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df
                
    def _remove_lowcase(self, se,min_count=5):
        count = dict(se.value_counts())
        se = se.map(lambda x : -1 if count[x]<min_count else x)
        return se
class HotCVStatistic2(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,kfold=10,is_test_use_val=True):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for concat_feat in self.feat_list:
            df_cv = df[['_index',concat_feat, self.label,'dataset_type']]
            new_feat_name = concat_feat + '_cvstat_hot'
            self.cvr_feat_list.append(new_feat_name)
          
             # train val test
            training = df_cv[df_cv['dataset_type']==-1]  
            training = training.reset_index(drop=True)
            valid = df_cv[df_cv.dataset_type==-2]  
            trainval = df_cv[df_cv['dataset_type']!=-3]
            trainval = trainval.reset_index(drop=True)
            predict = df_cv[df_cv.dataset_type==-3]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(kfold,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = self._statis(X_train,X_val,concat_feat,self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)
            
            X_valid = self._statis(training, valid, concat_feat,label=self.label)
            if is_test_use_val:
                X_pred = self._statis(trainval, predict, concat_feat,label=self.label)
            else:
                X_pred = self._statis(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_valid,X_pred],axis=0)#columns:(index,f,f_cvr,label)
            
#             X_pred = self._statis(training, predict, concat_feat,self.label)
#             df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del df_stas_feat['dataset_type']
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df
    def _statis(self,df,df_val, feature,label = 'label'):
        df = df.groupby(feature)[label].agg(['count']).reset_index()
        new_feat_name = feature + '_cvstat_hot'
        df.loc[:,new_feat_name] = df['count']/df.shape[0]
        df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
        df_stas = df[[feature,new_feat_name]]
        df_val = pd.merge(df_val, df_stas, how='left', on=feature)
        return df_val
      
class CrossHotCVStatistic2(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,kfold=10,is_test_use_val=True):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2 in self.feat_list:
            concat_feat = feat1+'_'+feat2
            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)
            df_cv = df[['_index',concat_feat, self.label,'dataset_type']]
            new_feat_name = concat_feat + '_cvstat_hot'
            self.cvr_feat_list.append(new_feat_name)
          
             # train val test
            training = df_cv[df_cv['dataset_type']==-1]  
            training = training.reset_index(drop=True)
            valid = df_cv[df_cv.dataset_type==-2]  
            trainval = df_cv[df_cv['dataset_type']!=-3]
            trainval = trainval.reset_index(drop=True)
            predict = df_cv[df_cv.dataset_type==-3]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(kfold,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = self._statis(X_train,X_val,concat_feat,self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_valid = self._statis(training, valid, concat_feat,label=self.label)
            if is_test_use_val:
                X_pred = self._statis(trainval, predict, concat_feat,label=self.label)
            else:
                X_pred = self._statis(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_valid,X_pred],axis=0)#columns:(index,f,f_cvr,label))

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del df_stas_feat['dataset_type']
            del df[concat_feat]
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df
    def _statis(self,df,df_val, feature,label = 'label'):
        df = df.groupby(feature)[label].agg(['count']).reset_index()
        new_feat_name = feature + '_cvstat_hot'
        df.loc[:,new_feat_name] = df['count']/df.shape[0]
        df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
        df_stas = df[[feature,new_feat_name]]
        df_val = pd.merge(df_val, df_stas, how='left', on=feature)
        return df_val


# In[12]:


df_data['prefix_title_tag'] = 'f1-'+df_data['prefix'].astype(str)+'_f2-'+df_data['title'].astype(str)+'_f3-'+df_data['tag'].astype(str)
df_data['prefix_num_title_num_tag'] = 'f1-'+df_data['prefix_num'].astype(str)+'_f2-'+df_data['title_num'].astype(str)+'_f3-' + df_data['tag'].astype(str)


# In[13]:


cvr_features = ['prefix', 'title', 'tag',
                'complete_prefix','prefix_title_tag','prefix_num_title_num_tag']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=5,is_test_use_val=False)

cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag'),
                ('complete_prefix','title'),('complete_prefix','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=5,is_test_use_val=False)
    
features = ['prefix', 'title', 'tag',
            'complete_prefix','prefix_title_tag','prefix_num_title_num_tag']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=5,is_test_use_val=False)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag'),
                ('complete_prefix','title'),('complete_prefix','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=5,is_test_use_val=False)


# # do cvr 5fold

# In[23]:


# cvr_features = ['prefix', 'title', 'tag',
#                 'complete_prefix']
# cvstor=CVRCVStatistic2(cvr_features,'label')
# with utils.timer('CVR'):
#     df_data = cvstor.cvs(df_data,0,kfold=5,is_test_use_val=False)

# cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag'),
#                 ('complete_prefix','title'),('complete_prefix','tag')]
# cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
# with utils.timer('CVR'):
#     df_data = cvstor.cvs(df_data,0,kfold=5,is_test_use_val=False)
    
# features = ['prefix', 'title', 'tag',
#             'complete_prefix']
# cvstor=HotCVStatistic2(features,'label')
# with utils.timer('Hot'):
#     df_data = cvstor.cvs(df_data,kfold=5,is_test_use_val=False)
# cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag'),
#                 ('complete_prefix','title'),('complete_prefix','tag')]
# cvstor=CrossHotCVStatistic2(cvr_features,'label')
# with utils.timer('Hot'):
#     df_data = cvstor.cvs(df_data,kfold=5,is_test_use_val=False)


# In[24]:


df_data.columns.values


# In[10]:


df_data = df_data.drop(columns=['prefix', 'query_prediction', 'title', 'tag', 'label',
       'dataset_type', 'max_similar', 'mean_similar', 'weight_similar',
       'is_in_title', 'leven_distance', 'distance_rate', 'prefix_w2v',
       'words', 'title_pred_score', 'title_unseen_nword', 'pred_freq_std',
       'pred_freq_mean', 'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
       'complete_prefix', 'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max'])


# In[11]:


df_data.to_csv('cvr_df.csv',index=False)


# # fillna and training

# In[14]:


for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


# In[15]:


with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)

    df_data = df_data.fillna(0.0)


# In[16]:


features= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']
features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]
save_dir = '/home/admin/jupyter/Demo/yao/data/features/'

df_text_train = pd.read_csv(save_dir+'train_text_match.csv')
df_text_valid = pd.read_csv(save_dir+'valid_text_match.csv')
df_text_testa = pd.read_csv(save_dir+'test_text_match.csv')
df_text = pd.concat([df_text_train,df_text_valid,df_text_testa]).reset_index(drop=True)


# In[30]:


df_text['texts'].head()


# In[17]:


df_text_train['words'] = df_text_train['texts'].apply(lambda x: x.split(' '))
df_text_valid['words'] = df_text_valid['texts'].apply(lambda x: x.split(' '))
df_text_testa['words'] = df_text_testa['texts'].apply(lambda x: x.split(' '))


# In[18]:


with utils.timer('CountVectorizer'):
    cvor = CountVectorizer(analyzer=lambda x: x,binary=True)
    cv_train = cvor.fit_transform(df_text_train['words']) 
    cv_valid = cvor.transform(df_text_valid['words']) 
    cv_testa = cvor.transform(df_text_testa['words']) 


# In[18]:


cv_train.shape


# In[19]:


del df_text_train,df_text_valid,df_text_testa
gc.collect()


# In[20]:


def clean_csr_2(csr_trn,  min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    return csr_trn[:, trn_min], trn_min
with utils.timer('clean'):
    cv_train,mask = clean_csr_2(cv_train,2)
    cv_valid = cv_valid[:,mask]
    cv_testa = cv_testa[:,mask]


# In[35]:


cv_train.shape


# In[21]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)


# In[23]:


# with utils.timer('CountVectorizer'):
#     cvor = CountVectorizer(analyzer=lambda x: x,binary=True)
#     cv_train = cvor.fit_transform(df_train['words']) 
#     cv_valid = cvor.transform(df_valid['words']) 
#     cv_testa = cvor.transform(df_testa['words']) 


# In[24]:


# df_train['words'].head()


# In[22]:


lb_enc = LabelEncoder()
enc = OneHotEncoder()
with utils.timer('Encoder'):
    tag_train = lb_enc.fit_transform(df_train['tag'])
    tag_valid = lb_enc.transform(df_valid['tag'])
    tag_testa = lb_enc.transform(df_testa['tag'])

    tag_train = enc.fit_transform(tag_train.reshape(-1, 1))
    tag_valid = enc.transform(tag_valid.reshape(-1, 1))
    tag_testa = enc.transform(tag_testa.reshape(-1, 1))


# In[46]:


# tag_train.shape


# In[38]:


df_data.columns.values


# # classification

# In[61]:


# features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
# features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
# # features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
# features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

# features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
#            'nword_title_unseen_in_prefix' ]
features = ['max_similar',
       'mean_similar', 'weight_similar', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'title_pred_score',
       'title_unseen_nword', 'pred_freq_std', 'pred_freq_mean',
       'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# # old results

# In[21]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# In[27]:


y_prob = clf.predict_proba(valid_x)
y_prob = y_prob[:,1]
y_prob_test = clf.predict_proba(testa_x)
y_prob_test = y_prob_test[:,1]
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # now

# In[63]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100,verbose=10)


# In[64]:


y_prob = clf.predict_proba(valid_x)
y_prob = y_prob[:,1]
y_prob_test = clf.predict_proba(testa_x)
y_prob_test = y_prob_test[:,1]
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # loss

# In[65]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric="logloss",early_stopping_rounds=100,verbose=10)


# In[66]:


y_prob = clf.predict_proba(valid_x)
y_prob = y_prob[:,1]
y_prob_test = clf.predict_proba(testa_x)
y_prob_test = y_prob_test[:,1]
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # add feat1

# In[26]:


def levenshtein_distance(str1,str2):
    if not isinstance(str1, str):
        str1 = "null"

    x_size = len(str1) + 1
    y_size = len(str2) + 1

    matrix = np.zeros((x_size, y_size), dtype=np.int_)

    for x in range(x_size):
        matrix[x, 0] = x

    for y in range(y_size):
        matrix[0, y] = y

    for x in range(1, x_size):
        for y in range(1, y_size):
            if str1[x - 1] == str2[y - 1]:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
            else:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)

    return matrix[x_size - 1, y_size - 1]
def levenshtein_distance_score(str1,str2):
    dist = levenshtein_distance(str1,str2)
    return dist/max(len(str1),len(str2))


# In[27]:


def title_preds_match_freq_rate(x):
    preds = eval(x['query_prediction'])
    
    if len(preds)>0:
        dist_scores = []
        freqs = []
        for pred,freq in preds.items():
            dist_scores.append(levenshtein_distance_score(x['title'],pred))
            freqs.append(float(freq))
        match_index = dist_scores.index(min(dist_scores))
        if max(freqs) == 0.0:
            return 0.0
        # 1
        return freqs[match_index]/max(freqs)
        # etc rank

    else:
        return 0.0
def title_preds_rank(x):
    preds = eval(x['query_prediction'])
    
    if len(preds)>0:
        dist_scores = []
        freqs = []
        for pred,freq in preds.items():
            dist_scores.append(levenshtein_distance_score(x['title'],pred))
            freqs.append(float(freq))
        match_index = dist_scores.index(min(dist_scores))

        # 1
        obj = pd.Series(freqs)
        return obj.rank(method = 'min',ascending=False)[match_index]
        # etc rank
    else:
        return 11.0


# In[28]:


with utils.timer('title_preds_match_freq_rate'):
    df_data['title_preds_match_freq_rate'] = df_data.apply(lambda x: title_preds_match_freq_rate(x),axis = 1)


# In[29]:


with utils.timer('title_preds_match_freq_rate'):
    df_data['title_preds_rank'] = df_data.apply(lambda x: title_preds_rank(x),axis = 1)


# In[30]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)


# In[31]:


features = ['max_similar', 'mean_similar', 'weight_similar',
       'is_in_title', 'leven_distance', 'distance_rate', 'prefix_w2v',
       'title_pred_score', 'title_unseen_nword', 'pred_freq_std',
       'pred_freq_mean', 'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
      'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'cate_prefix_cvstat_cvr', 'cate_title_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'cate_prefix_cate_title_cvstat_cvr', 'cate_prefix_tag_cvstat_cvr',
       'cate_title_tag_cvstat_cvr', 'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'cate_prefix_cvstat_hot',
       'cate_title_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'cate_prefix_cate_title_cvstat_hot', 'cate_prefix_tag_cvstat_hot',
       'cate_title_tag_cvstat_hot', 'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']
features +=['title_preds_match_freq_rate','title_preds_rank']

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[32]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8,verbose=10
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# In[33]:


y_prob = clf.predict_proba(valid_x)
y_prob = y_prob[:,1]
y_prob_test = clf.predict_proba(testa_x)
y_prob_test = y_prob_test[:,1]
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # Regression

# In[67]:


df_train_score = df_train.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_valid_score = df_valid.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_train_reg = pd.merge(df_train,df_train_score,on=['prefix','tag','title'],how='left')
df_valid_reg = pd.merge(df_valid,df_valid_score,on=['prefix','tag','title'],how='left')
df_train_reg['id'] = df_train_reg.index
df_valid_reg['id'] = df_valid_reg.index
df_train_reg = df_train_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
df_valid_reg = df_valid_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
train_keep_indices = df_train_reg['id'].values
valid_keep_indices = df_valid_reg['id'].values


# In[68]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]


# In[69]:


# features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
# features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
# # features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
# features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

# features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
#            'nword_title_unseen_in_prefix' ]
features = ['max_similar',
       'mean_similar', 'weight_similar', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'title_pred_score',
       'title_unseen_nword', 'pred_freq_std', 'pred_freq_mean',
       'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']

with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))


# In[70]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100,verbose=10)


# In[32]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[33]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=10)


# In[34]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[62]:


testa_x.shape


# # bayes

# In[71]:


# https://github.com/LightR0/Tencent_Ads_2018
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)
def bayes_ctr(df,feature,label = 'label'):
    df = df.groupby(feature)[label].agg(['sum','count']).reset_index()
    new_feat_name = feature + '_bayes_ctr'
    I = df['count']
    C = df['sum']
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(I,C , 1000, 0.00000001)
    df.loc[:,new_feat_name] = (C+hyper.alpha)/(I+hyper.alpha+hyper.beta)
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
    df_stas = df[[feature,new_feat_name]]
    return df_stas
def cross_bayes_ctr(df,features,label = 'label'):
    df = df.groupby(features)[label].agg(['sum','count']).reset_index()
    feature = ''
    for idx,feat in enumerate(features):
        if idx == 0:
            feature += feat
        else:
            feature += '_'+feat
    new_feat_name = feature + '_bayes_ctr'
    I = df['count']
    C = df['sum']
    hyper = HyperParam(1, 1)
    hyper.update_from_data_by_FPI(I,C , 1000, 0.00000001)
    df.loc[:,new_feat_name] = (C+hyper.alpha)/(I+hyper.alpha+hyper.beta)
    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
    df_stas = df[features+[new_feat_name]]
    return df_stas


# In[72]:


df_train_score = cross_bayes_ctr(df_train,['prefix','tag','title']).rename(columns={'prefix_tag_title_bayes_ctr':'score'})
df_valid_score = df_valid.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_train_reg = pd.merge(df_train,df_train_score,on=['prefix','tag','title'],how='left')
df_valid_reg = pd.merge(df_valid,df_valid_score,on=['prefix','tag','title'],how='left')
df_train_reg['id'] = df_train_reg.index
df_valid_reg['id'] = df_valid_reg.index
df_train_reg = df_train_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
df_valid_reg = df_valid_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
train_keep_indices = df_train_reg['id'].values
valid_keep_indices = df_valid_reg['id'].values


# In[37]:


# df_train.head()


# In[73]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]


# In[74]:


features = ['max_similar',
       'mean_similar', 'weight_similar', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'title_pred_score',
       'title_unseen_nword', 'pred_freq_std', 'pred_freq_mean',
       'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']

with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))


# In[99]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)],early_stopping_rounds=100,verbose=10)


# In[75]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)],early_stopping_rounds=100,verbose=10)


# In[76]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(reg_testa_x)
for thr in np.linspace(0.35,0.55,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # ALL DATA

# In[77]:


df_trainval = pd.concat([df_train,df_valid],ignore_index=True)


# In[78]:


df_trainval.shape


# In[79]:


df_train_score = cross_bayes_ctr(df_trainval,['prefix','tag','title']).rename(columns={'prefix_tag_title_bayes_ctr':'score'})
df_valid_score = df_train_score
df_train_reg = pd.merge(df_trainval,df_train_score,on=['prefix','tag','title'],how='left')
df_valid_reg = df_train_reg
df_train_reg['id'] = df_train_reg.index
df_valid_reg['id'] = df_valid_reg.index
df_train_reg = df_train_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
df_valid_reg = df_valid_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
train_keep_indices = df_train_reg['id'].values
valid_keep_indices = df_valid_reg['id'].values


# In[80]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values


# In[81]:


features = ['max_similar',
       'mean_similar', 'weight_similar', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'title_pred_score',
       'title_unseen_nword', 'pred_freq_std', 'pred_freq_mean',
       'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']

with utils.timer('Stack Feature'):
    cv_trainval = sparse.vstack((cv_train,cv_valid))
    tag_trainval = sparse.vstack((tag_train,tag_valid))
    reg_train_x= sparse.hstack((cv_trainval,tag_trainval)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = reg_train_x
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = reg_train_x
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))


# In[84]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=2260, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=10)


# In[96]:


# y_prob = clf.predict(valid_x)
# y_prob_test = clf.predict(reg_testa_x)
# for thr in np.linspace(0.35,0.65,40):
#     pred = (y_prob>thr).astype(np.uint32)
#     pred_test = (y_prob_test>thr).astype(np.uint32)
#     f1 = f1_score(valid_y, pred)
#     mean_pred=sum(pred)/len(pred)
#     mean_pred_test = sum(pred_test)/len(pred_test)
#     print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[101]:


y_prob = clf.predict(testa_x)


# In[88]:


y_prob = pd.DataFrame(y_prob)
y_prob.to_csv('prob_reg_7465.csv',index=False,header=False)


# In[102]:


y_pred = (y_prob>0.42).astype(np.uint32)


# In[103]:


y_pred.mean()


# In[104]:


y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('pred_reg_7465_0.42.csv',index=False,header=False)


# In[42]:


reg_train_x.shape


# In[47]:


y_prob_test = pd.DataFrame()


# In[48]:


y_prob_test.to_csv('prob_reg_7465.csv',index=False,header=False)


# In[108]:


pred_test = (y_prob_test>0.4192 ).astype(np.uint32)


# In[109]:


pred_test.shape


# In[110]:


pred_test = pd.DataFrame(pred_test)


# In[111]:


pred_test.to_csv('predict1125.csv',index=False,header=False)


# In[112]:


import zipfile
a = zipfile.ZipFile('test1125.zip','w')
a.write("predict1125.csv")
a.close()


# In[113]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(valid_x, valid_y)],early_stopping_rounds=100,verbose=10)


# In[114]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(reg_testa_x)
for thr in np.linspace(0.35,0.65,40):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[54]:


df_train.shape


# In[55]:


df_valid.shape


# # 特征选择

# In[50]:


from featexp import get_trend_stats
from featexp import get_univariate_plots
get_univariate_plots(data = df_train, target_col='label',                      data_test = df_valid, features_list=['prob_max'],bins=10)


# In[57]:


stats = get_trend_stats(data=df_train, target_col='label',                      data_test=df_valid)


# In[58]:


stats[stats.Trend_correlation < 0.5]


# In[49]:


del train_x,valid_x,testa_x
gc.collect()


# # drop some cvr

# In[ ]:


# title_tag_cvstat_hot , title_cvstat_hot, prob_max , pred_freq_std


# In[59]:


features = ['max_similar',
       'mean_similar', 'weight_similar', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'title_pred_score',
       'title_unseen_nword', 'pred_freq_std', 'pred_freq_mean',
       'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 
       'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']

# 'cate_prefix_cvstat_cvr', 'cate_title_cvstat_cvr',
#        'complete_prefix_cvstat_cvr', 
#        'prefix_num_title_num_tag_cvstat_cvr','cate_prefix_cate_title_cvstat_cvr', 'cate_prefix_tag_cvstat_cvr',
#         'cate_title_tag_cvstat_cvr', 'complete_prefix_title_cvstat_cvr',
#         'complete_prefix_tag_cvstat_cvr',prefix_title_tag_cvstat_cvr',
# 'cate_prefix_cvstat_hot',
#        'cate_title_cvstat_hot', 'complete_prefix_cvstat_hot',
#        'prefix_num_title_num_tag_cvstat_hot','cate_prefix_cate_title_cvstat_hot', 'cate_prefix_tag_cvstat_hot',
#        'cate_title_tag_cvstat_hot', 'complete_prefix_title_cvstat_hot',
#        'complete_prefix_tag_cvstat_hot'
        
with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[60]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100,verbose=10)


# In[43]:


y_prob = clf.predict_proba(valid_x)
y_prob = y_prob[:,1]
y_prob_test = clf.predict_proba(testa_x)
y_prob_test = y_prob_test[:,1]
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[116]:


df_train_score = cross_bayes_ctr(df_train,['prefix','tag','title']).rename(columns={'prefix_tag_title_bayes_ctr':'score'})
df_valid_score = df_valid.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_train_reg = pd.merge(df_train,df_train_score,on=['prefix','tag','title'],how='left')
df_valid_reg = pd.merge(df_valid,df_valid_score,on=['prefix','tag','title'],how='left')
df_train_reg['id'] = df_train_reg.index
df_valid_reg['id'] = df_valid_reg.index
df_train_reg = df_train_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
df_valid_reg = df_valid_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
train_keep_indices = df_train_reg['id'].values
valid_keep_indices = df_valid_reg['id'].values


# In[117]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]


# In[118]:


features = ['max_similar', 'mean_similar', 'weight_similar',
       'is_in_title', 'leven_distance', 'distance_rate', 'prefix_w2v',
       'title_pred_score', 'title_unseen_nword', 'pred_freq_std',
       'pred_freq_mean', 'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
      'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
        'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 
        'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot']
with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))


# In[119]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)],early_stopping_rounds=100,verbose=10)


# In[120]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[ ]:


pred_test = (y_prob_test>0.4192 ).astype(np.uint32)


# In[ ]:





# # no countvector

# In[86]:


features = ['max_similar', 'mean_similar', 'weight_similar',
       'is_in_title', 'leven_distance', 'distance_rate', 'prefix_w2v',
       'title_pred_score', 'title_unseen_nword', 'pred_freq_std',
       'pred_freq_mean', 'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'cate_prefix', 'cate_title', 'prefix_num', 'title_num',
      'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'max_query_ratio',
       'title_word_num', 'small_query_num', 'prob_max',
       'prefix_cvstat_cvr', 'title_cvstat_cvr', 'tag_cvstat_cvr',
       'cate_prefix_cvstat_cvr', 'cate_title_cvstat_cvr',
       'complete_prefix_cvstat_cvr', 'prefix_title_tag_cvstat_cvr',
       'prefix_num_title_num_tag_cvstat_cvr', 'prefix_title_cvstat_cvr',
       'prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr',
       'cate_prefix_cate_title_cvstat_cvr', 'cate_prefix_tag_cvstat_cvr',
       'cate_title_tag_cvstat_cvr', 'complete_prefix_title_cvstat_cvr',
       'complete_prefix_tag_cvstat_cvr', 'prefix_cvstat_hot',
       'title_cvstat_hot', 'tag_cvstat_hot', 'cate_prefix_cvstat_hot',
       'cate_title_cvstat_hot', 'complete_prefix_cvstat_hot',
       'prefix_title_tag_cvstat_hot',
       'prefix_num_title_num_tag_cvstat_hot', 'prefix_title_cvstat_hot',
       'prefix_tag_cvstat_hot', 'title_tag_cvstat_hot',
       'cate_prefix_cate_title_cvstat_hot', 'cate_prefix_tag_cvstat_hot',
       'cate_title_tag_cvstat_hot', 'complete_prefix_title_cvstat_hot',
       'complete_prefix_tag_cvstat_hot']


# In[52]:


# df_data.to_csv('df_cvrk10.csv',index=False)


# In[87]:


df_data['tag'] = LabelEncoder().fit_transform(df_data['tag'])


# In[95]:


df_data.shape


# In[88]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)


# In[89]:


# df_train_x = df_train[features]
valid_x = df_valid[features]
testa_x = df_testa[features]


# In[ ]:


# df_train_y = df_train['label'].apply(int)
# df_valid_y = df_train['label'].apply(int)


# In[90]:


df_train_score = df_train.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_valid_score = df_valid.groupby(['prefix','tag','title'])['label'].apply(lambda x: sum(x)/len(x)).reset_index(drop = False).rename(columns={'label':'score'})
df_train_reg = pd.merge(df_train,df_train_score,on=['prefix','tag','title'],how='left')
df_valid_reg = pd.merge(df_valid,df_valid_score,on=['prefix','tag','title'],how='left')
df_train_reg['id'] = df_train_reg.index
df_valid_reg['id'] = df_valid_reg.index
df_train_reg = df_train_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
df_valid_reg = df_valid_reg.drop_duplicates(['prefix','tag','title'],keep = 'first')
train_keep_indices = df_train_reg['id'].values
valid_keep_indices = df_valid_reg['id'].values


# In[91]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]


# In[92]:


reg_train_x = df_train_reg[features]
reg_valid_x = df_valid_reg[features]


# In[93]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=10)


# In[94]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[ ]:




