
#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import string
import unicodedata
from itertools import groupby
import re
from collections import defaultdict
import scipy.special as special


# In[2]:


def my_f1_score(labels, preds):
    return 'f1_score', f1_score(labels,preds.round()), True


# In[3]:


root_dir = '../../DataSets/oppo_data_ronud2_20181107/'


# In[4]:


df_valid = utils.read_txt(root_dir+'data_vali.txt')
df_train = utils.read_txt(root_dir+'data_train.txt')
df_testa = utils.read_txt(root_dir+'data_testb.txt',is_label=False)
#
df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)
train_y = df_train['label'].astype(int).values
valid_y = df_valid['label'].astype(int).values
#
df_testa['label']=-1
df_train['dataset_type']=-1
df_valid['dataset_type']=-2
df_testa['dataset_type']=-3
df_data=pd.concat([df_train,df_valid,df_testa]).reset_index(drop=True)


# # # text clean

# In[ ]:


def text_clean(text,patterns=[]):
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    """ Simple text clean up process"""
    #  1. Go to lower case (only good for english)
    clean = text.lower()
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return clean


# In[ ]:


cont_pattern = [('','')]
df_data['prefix'] = df_data['prefix'].apply(lambda x:text_clean(x,cont_pattern))
df_data['title'] = df_data['title'].apply(lambda x:text_clean(x,cont_pattern))


# In[ ]:


df_data['query_prediction'] = df_data['query_prediction'].apply(lambda x:x if x!='' else '{}')
df_data['query_prediction'] = df_data['query_prediction'].apply(lambda x:text_clean(x,cont_pattern))


# In[ ]:


df_data['cut_title'] = df_data['title'].apply(lambda x: jieba.lcut(x))
df_data['cut_prefix'] = df_data['prefix'].apply(lambda x: jieba.lcut(x))


# In[ ]:


def func(x):
    preds = x['query_prediction'].unique()
    preds = [_x for _x in preds if len(_x)>0]
    
    if len(preds)>1:
#         print(preds)
        converted_preds = [eval(pred) for pred in preds]
        preds_num = [len(pred) for pred in converted_preds]
        max_index = preds_num.index(max(preds_num))
#         print(preds[max_index])
        return preds[max_index]
    if len(preds)==0:
        return ''
    else:
        return preds[0]
with utils.timer('pad query_predictio'): 
    df_tmp = df_data.groupby(['prefix']).apply(lambda x:func(x)).reset_index(drop=False).rename(columns={0:'query_prediction'})
    del df_data['query_prediction']
    df_data = pd.merge(df_data,df_tmp,on='prefix',how='left')


# # stat

# In[ ]:


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
#                 print(concat_feat+' split order: ',cnt)
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
#                 print(concat_feat+' split order: ',cnt)
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
#                 print(concat_feat+' split order: ',cnt)
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
#                 print(concat_feat+' split order: ',cnt)
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


# In[ ]:


cvr_features = ['prefix', 'title', 'tag']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=5)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=5)


# In[ ]:


features = ['prefix', 'title']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=5)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=5)


# # text match feature

# In[ ]:


with utils.timer('query_predictio jieba cut'): 
    df_data['cut_query_prediction'] = df_data['query_prediction'].apply(lambda x:{k:[[float(v)],jieba.lcut(k)] for k,v in eval(x).items()})


# In[ ]:


def func(x):
    text = []
    text.extend(x['cut_prefix'])
    text.extend(x['cut_title'])
    preds=  []
    for k,v in x['cut_query_prediction'].items():
        preds.extend(v[1] )
    text.extend(preds)
    return text
with utils.timer('gen text'):
    df_data['words']=df_data.apply(lambda x:func(x),axis=1)


# In[ ]:


def func(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    std = np.std(freqs) if len(freqs)>0 else 0.0
    return std
with utils.timer('pred_freq_std'):
    df_data['pred_freq_std']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    std = np.mean(freqs) if len(freqs)>0 else 0.0
    return std
with utils.timer('pred_freq_mean'):
    df_data['pred_freq_mean']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    query_prediction = x['cut_query_prediction']
    freqs = []
    for k,v in query_prediction.items():
        freqs.extend(v[0])
    _sum = np.sum(freqs) if len(freqs)>0 else 0.0
    return _sum
with utils.timer('pred_freq_sum'):
    df_data['pred_freq_sum']=df_data.apply(lambda x:func(x),axis=1)


# In[ ]:


def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            score +=v[0][0]
    return score
with utils.timer('title_pred_score'):
    df_data['title_pred_score']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    fit_nwords=[]
    fit_keys = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            fit_nwords.append(len(pred_words))
            fit_keys.append(k)
    if len(fit_keys)>0:
        k = fit_keys[np.argmax(fit_nwords)]
        score = query_prediction[k][0][0]
    return score
with utils.timer('title_unseen_in_prefix_score_max'):
    df_data['title_unseen_in_prefix_score_max']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    
    scores = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            scores.append(v[0])
    std = np.std(scores) if len(scores)>0 else 0.0
    return std
with utils.timer('title_unseen_in_prefix_score_std'):
    df_data['title_unseen_in_prefix_score_std']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    score = 0.0
    
    scores = []
    for k,v in query_prediction.items():
        pred_words = v[1]
        pred_words = [word for word in pred_words if word in title_words]
        if len(pred_words)>0:
            scores.append(v[0])
    mean = np.mean(scores) if len(scores)>0 else 0.0
    return mean
with utils.timer('title_unseen_in_prefix_score_mean'):
    df_data['title_unseen_in_prefix_score_mean']=df_data.apply(lambda x:func(x),axis=1)


# In[ ]:


def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    return len(title_words)
with utils.timer('nword_title_unseen_in_prefix'):
    df_data['nword_title_unseen_in_prefix']=df_data.apply(lambda x:func(x),axis=1)
def func(x):
    words = []
    prefix_words = x['cut_prefix']
    title_words = x['cut_title']
    title_words = [word for word in title_words if word not in prefix_words]
    query_prediction = x['cut_query_prediction']
    nwords = 0.0
    new_words= []
    pred_words = []
    for k,v in query_prediction.items():
        pred_words.extend(v[1])
    new_words=[word for word in title_words if word not in pred_words]
    return len(new_words)
with utils.timer('title_unseen_nword'):
    df_data['title_unseen_nword']=df_data.apply(lambda x:func(x),axis=1)
with utils.timer('prefix_nwords'):
    df_data['prefix_nwords']=df_data['cut_prefix'].apply(lambda x: len(x))
with utils.timer('title_nwords'):
    df_data['title_nwords']=df_data['cut_title'].apply(lambda x: len(x))


# In[ ]:


features= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum','title_nwords']
features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]
df_data['texts'] = df_data['words'].apply(lambda x: ' '.join(x))
features.append('texts')
save_dir = '../data/cleaned_features/'
os.makedirs(save_dir,exist_ok=True)
df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)
df_train[features].to_csv(save_dir+'train_text_match.csv',index=False)
df_valid[features].to_csv(save_dir+'valid_text_match.csv',index=False)
df_testa[features].to_csv(save_dir+'test_text_match.csv',index=False)


# # load text feature

# In[13]:


features= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']
features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]
save_dir = '../data/cleaned_features/'

df_text_train = pd.read_csv(save_dir+'train_text_match.csv')
df_text_valid = pd.read_csv(save_dir+'valid_text_match.csv')
df_text_testa = pd.read_csv(save_dir+'test_text_match.csv')
df_text = pd.concat([df_text_train,df_text_valid,df_text_testa]).reset_index(drop=True)


# In[14]:


df_data = pd.concat([df_data.reset_index(drop=True),df_text],axis=1)
df_data['words'] = df_data['texts'].apply(lambda x: x.split(' '))


# # text

# In[ ]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)
with utils.timer('CountVectorizer'):
    cvor = CountVectorizer(analyzer=lambda x: x,binary=True)
    cv_train = cvor.fit_transform(df_train['words']) 
    cv_valid = cvor.transform(df_valid['words']) 
    cv_testa = cvor.transform(df_testa['words']) 


# In[ ]:


lb_enc = LabelEncoder()
enc = OneHotEncoder()
with utils.timer('Encoder'):
    tag_train = lb_enc.fit_transform(df_train['tag'])
    tag_valid = lb_enc.transform(df_valid['tag'])
    tag_testa = lb_enc.transform(df_testa['tag'])

    tag_train = enc.fit_transform(tag_train.reshape(-1, 1))
    tag_valid = enc.transform(tag_valid.reshape(-1, 1))
    tag_testa = enc.transform(tag_testa.reshape(-1, 1))


# In[ ]:


def clean_csr_2(csr_trn,  min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    return csr_trn[:, trn_min], trn_min
with utils.timer('clean'):
    cv_train,mask = clean_csr_2(cv_train,50)
    cv_valid = cv_valid[:,mask]
    cv_testa = cv_testa[:,mask]


# # feature clean

# In[ ]:


with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)


# In[27]:


from featexp import get_trend_stats,get_univariate_plots


# In[28]:


stats = get_trend_stats(data=df_data[df_data['dataset_type']==-1], target_col='label',                      data_test=df_data[df_data['dataset_type']==-2])


# In[30]:


stats[stats['Trend_correlation']<0.8]


# In[46]:


get_univariate_plots(data=df_data[df_data['dataset_type']==-1], target_col='label',                      data_test=df_data[df_data['dataset_type']==-2], features_list=['prefix_tag_cvstat_hot'],bins=10)


# In[19]:


features = ['title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']+['title_nwords']
features+= ['prefix_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot']
features+= ['title_unseen_nword', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[22]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,verbose=10,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100,verbose=10)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


features = ['title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']+['title_nwords']
features+= ['prefix_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot']
features+= ['title_unseen_nword', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))


# In[ ]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]


# In[ ]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=50)


# In[28]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[29]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=50)


# In[30]:


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# # all data

# In[31]:


df_trainval = pd.concat([df_train,df_valid],ignore_index=True)


# In[32]:


df_trainval.shape


# In[33]:


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


# In[34]:


reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values


# In[35]:


features = ['title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']+['title_nwords']
features+= ['prefix_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot']
features+= ['title_unseen_nword', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

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


# In[37]:


clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=1780, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.1, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100,verbose=100)


# In[70]:


# # y_pred = clf.predict(testa_x)
# thr = 0.4263
# y_prob = clf.predict(testa_x)
# save_dir = '../tmp/lgb_ml_all_reg_1201/submit/'
# os.makedirs(save_dir,exist_ok=True)

# with open(save_dir+'lgb_ml_all_reg_1201'+str(thr)+'.csv.prob','w') as fout:
#     for prob in y_prob:
#         fout.write(str(prob)+'\n')
        
# y_pred = (y_prob>thr).astype(np.uint32)
# print(y_pred.sum()/len(y_pred))

# with open(save_dir+'lgb_ml_all_reg_1129'+str(thr)+'.csv','w') as fout:
#     for pred in y_pred:
#         fout.write(str(pred)+'\n')


# In[38]:


# y_pred = clf.predict(testa_x)
thr = 0.40
y_prob = clf.predict(testa_x)
save_dir = '../tmp/lgb_ml_all_reg_1202/submit/'
os.makedirs(save_dir,exist_ok=True)

with open(save_dir+'lgb_ml_all_reg_1202'+str(thr)+'.csv.prob','w') as fout:
    for prob in y_prob:
        fout.write(str(prob)+'\n')
        
y_pred = (y_prob>thr).astype(np.uint32)
print(y_pred.sum()/len(y_pred))

with open(save_dir+'lgb_ml_all_reg_1202'+str(thr)+'.csv','w') as fout:
    for pred in y_pred:
        fout.write(str(pred)+'\n')


# In[ ]:


print("ok")


# In[ ]:




