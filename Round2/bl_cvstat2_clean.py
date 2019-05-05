
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


# In[2]:


def my_f1_score(labels, preds):
    return 'f1_score', f1_score(labels,preds.round()), True


# In[3]:


root_dir = '../../DataSets/oppo_data_ronud2_20181107/'


# In[4]:


df_valid = utils.read_txt(root_dir+'data_vali.txt')
df_train = utils.read_txt(root_dir+'data_train.txt')
df_testa = utils.read_txt(root_dir+'data_test.txt',is_label=False)
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


# # text clean

# In[5]:


def text_clean(text,patterns=[]):
    patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    """ Simple text clean up process"""
    #  1. Go to lower case (only good for english)
    clean = text.lower()
    # 3. Replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    return clean
cont_pattern = [('','')]
text_clean('西瓜%2C',cont_pattern)


# In[6]:


df_data['cleaned_prefix'] = df_data['prefix'].apply(lambda x:text_clean(x,cont_pattern))


# In[7]:


cont_pattern = [('','')]
df_data['cleaned_title'] = df_data['title'].apply(lambda x:text_clean(x,cont_pattern))
df_data['cut_title'] = df_data['cleaned_title'].apply(lambda x: jieba.lcut(x))


# In[ ]:


df_data['cut_prefix'] = df_data['cleaned_prefix'].apply(lambda x: jieba.lcut(x))


# In[8]:


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
df_tmp = df_data.groupby(['cleaned_prefix']).apply(lambda x:func(x)).reset_index(drop=False).rename(columns={0:'padded_query_prediction'})
df_data = pd.merge(df_data,df_tmp,on='cleaned_prefix',how='left')


# In[10]:


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


# # ngram prefix

# In[27]:


def get_tf(texts):
    tf = defaultdict(int)
    for doc in texts:
        for token in doc:
            tf[token] += 1
    return tf
def filter_tf(tf,min_tf):
    return dict((term, freq) for term, freq in tf.items()
                           if freq >= min_tf)


# In[29]:



def func(x):
    prefix_tf=get_tf(x['cut_prefix'].tolist())
#     filter_tf(prefix_tf,min(len(x),max(len(x)*0.1,5)))
    tf = filter_tf(prefix_tf,min(len(x),5))
    return tf
df_tmp = df_data.groupby(['title']).apply(lambda x: func(x)).reset_index(drop=False).rename(columns={0:'prefix_tf'})
df_data = pd.merge(df_data,df_tmp,on='title',how='left')
def func(x):
    prefix = [_x for _x in  x['cut_prefix'] if _x in x['prefix_tf'].keys()]
    prefix = ''.join(prefix)
    if prefix =='':
        return x['prefix']
    else:
        return prefix
df_data['filter_prefix'] = df_data.apply(lambda x: func(x),axis=1)


# # ngram title

# In[ ]:


def func(x):
    title_tf=get_tf(x['cut_title'].tolist())
    tf = filter_tf(title_tf,min(len(x),5))
    return tf
df_tmp = df_data.groupby(['prefix']).apply(lambda x: func(x)).reset_index(drop=False).rename(columns={0:'title_tf'})
df_data = pd.merge(df_data,df_tmp,on='prefix',how='left')
def func(x):
    title = [_x for _x in  x['cut_title'] if _x in x['title_tf'].keys()]
    title = ''.join(title)
    if title =='':
        return x['title']
    else:
        return title
df_data['filter_title'] = df_data.apply(lambda x: func(x),axis=1)


# # stat

# # text match feature

# In[11]:


features= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']
features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]
save_dir = '../data/features/'

df_text_train = pd.read_csv(save_dir+'train_text_match.csv')
df_text_valid = pd.read_csv(save_dir+'valid_text_match.csv')
df_text_testa = pd.read_csv(save_dir+'test_text_match.csv')
df_text = pd.concat([df_text_train,df_text_valid,df_text_testa]).reset_index(drop=True)


# In[12]:


df_data = pd.concat([df_data.reset_index(drop=True),df_text],axis=1)


# In[13]:


df_data['words'] = df_data['texts'].apply(lambda x: x.split(' '))


# In[14]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)

with utils.timer('CountVectorizer'):
    cvor = CountVectorizer(analyzer=lambda x: x,binary=True)
    cv_train = cvor.fit_transform(df_train['words']) 
    cv_valid = cvor.transform(df_valid['words']) 
    cv_testa = cvor.transform(df_testa['words']) 
def clean_csr_2(csr_trn,  min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    return csr_trn[:, trn_min], trn_min
with utils.timer('clean'):
    cv_train,mask = clean_csr_2(cv_train,2)
    cv_valid = cv_valid[:,mask]
    cv_testa = cv_testa[:,mask]
    
lb_enc = LabelEncoder()
enc = OneHotEncoder()
with utils.timer('Encoder'):
    tag_train = lb_enc.fit_transform(df_train['tag'])
    tag_valid = lb_enc.transform(df_valid['tag'])
    tag_testa = lb_enc.transform(df_testa['tag'])

    tag_train = enc.fit_transform(tag_train.reshape(-1, 1))
    tag_valid = enc.transform(tag_valid.reshape(-1, 1))
    tag_testa = enc.transform(tag_testa.reshape(-1, 1))


# # stat

# In[12]:


cvr_features = ['prefix', 'title', 'tag']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)


# In[13]:


# filter
cvr_features = ['filter_prefix']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)
cvr_features = [('filter_prefix','title'), ('filter_prefix','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)


# In[14]:


features = ['prefix', 'title']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)


# In[15]:


# filter
features = ['filter_prefix']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)
cvr_features = [('filter_prefix','title'), ('filter_prefix','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)


# # text match feature

# In[19]:


for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


# In[20]:


with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)
    # filter
    features = ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)


# # classification

# In[26]:


features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
features+= ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
features+= ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[27]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# In[28]:


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


# # title clean

# In[33]:


# filter
cvr_features = ['filter_title']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)
cvr_features = [('filter_prefix','filter_title'), ('prefix','filter_title'),('filter_title','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)


# In[35]:


features = ['filter_title']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)
cvr_features = [('filter_prefix','filter_title'), ('prefix','filter_title'),('filter_title','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10)


# In[36]:


for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


# In[ ]:


with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)
    # filter
    features = ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)
    # title filter
    features = ['filter_title_cvstat_cvr','filter_prefix_filter_title_cvstat_cvr','prefix_filter_title_cvstat_cvr','filter_title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)
    features = ['filter_title_cvstat_hot','filter_prefix_filter_title_cvstat_hot','prefix_filter_title_cvstat_hot','filter_title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)


# In[40]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)


# In[41]:


features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
features+= ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
features+= ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']

features+= ['filter_title_cvstat_cvr','filter_prefix_filter_title_cvstat_cvr','prefix_filter_title_cvstat_cvr','filter_title_tag_cvstat_cvr']
features+= ['filter_title_cvstat_hot','filter_prefix_filter_title_cvstat_hot','prefix_filter_title_cvstat_hot','filter_title_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[42]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


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


# In[46]:


features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
features+= ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
# features+= ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']

features+= ['filter_title_cvstat_cvr','filter_prefix_filter_title_cvstat_cvr','prefix_filter_title_cvstat_cvr','filter_title_tag_cvstat_cvr']
# features+= ['filter_title_cvstat_hot','filter_prefix_filter_title_cvstat_hot','prefix_filter_title_cvstat_hot','filter_title_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[47]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# # stat

# In[15]:


cvr_features = ['cleaned_prefix', 'cleaned_title', 'tag']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)
cvr_features = [('cleaned_prefix','cleaned_title'), ('cleaned_prefix','tag'),('cleaned_title','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10)


# In[17]:


# filter
cvr_features = ['cleaned_prefix', 'cleaned_title', 'tag']
cvstor=HotCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,kfold=10)
cvr_features = [('cleaned_prefix','cleaned_title'), ('cleaned_prefix','tag'),('cleaned_title','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,kfold=10)


# In[18]:


for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


# In[20]:


with utils.timer('fillna'):
    features = ['cleaned_prefix_cvstat_cvr','cleaned_title_cvstat_cvr','tag_cvstat_cvr','cleaned_prefix_cleaned_title_cvstat_cvr','cleaned_prefix_tag_cvstat_cvr', 'cleaned_title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['cleaned_prefix_cvstat_hot','cleaned_title_cvstat_hot','cleaned_prefix_cleaned_title_cvstat_hot','cleaned_prefix_tag_cvstat_hot','cleaned_title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)


# In[22]:


df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)

features = ['cleaned_prefix_cvstat_cvr','cleaned_title_cvstat_cvr','tag_cvstat_cvr','cleaned_prefix_cleaned_title_cvstat_cvr','cleaned_prefix_tag_cvstat_cvr','cleaned_title_tag_cvstat_cvr']
features+= ['cleaned_prefix_cvstat_hot','cleaned_title_cvstat_hot','cleaned_prefix_cleaned_title_cvstat_hot','cleaned_prefix_tag_cvstat_hot','cleaned_title_tag_cvstat_hot']
# features+= ['filter_prefix_cvstat_cvr','filter_prefix_title_cvstat_cvr','filter_prefix_tag_cvstat_cvr']
# features+= ['filter_prefix_cvstat_hot','filter_prefix_title_cvstat_hot','filter_prefix_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[23]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# In[24]:


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


# In[ ]:




