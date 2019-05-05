#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Reference:
# https://zhuanlan.zhihu.com/p/46479794 (tf nn 0.7 不完整 )
# https://github.com/GrinAndBear/OGeek?tdsourcetag=s_pctim_aiomsg (0.6643 lg)
# https://github.com/flytoylf/OGeek（keras rnn cnn lgb 0.7）
# https://zhuanlan.zhihu.com/p/46482521(lgb 0.7)
# https://github.com/luoling1993/TianChi_OGeek
# https://blog.csdn.net/wang_shen_tao/article/details/50682478(文本相似度特征)
# https://nbviewer.jupyter.org/github/lianzhibin/tianchi_oppo/blob/master/add_click_history.ipynb（73.6）
# https://zhuanlan.zhihu.com/p/47807544?utm_source=com.tencent.tim&utm_medium=social&utm_oi=555381879923224576
# https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.11409386.0.0.52ee1d07cZDltT&raceId=231688&postsId=34595


# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
import utils
import numpy as np
import pandas as pd
import seaborn as sns
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


# In[2]:


def my_f1_score(labels, preds):
    return 'f1_score', f1_score(labels,preds.round()), True

# def my_f1_score(labels, preds):
#     return 'f1_score', f1_score(labels,np.where(preds>0.4, 1,0)), True

# In[3]:


root_dir = '/home/lab-xu.tianyuan/code/ogeek/'
data_dir = root_dir+'data/RawData/'


# In[4]:


df_valid = utils.read_txt(data_dir+'oppo_round1_vali.txt')
df_train = utils.read_txt(data_dir+'oppo_round1_train.txt')
df_testa = utils.read_txt(data_dir+'oppo_round1_test_A.txt',is_label=False)


# In[5]:


df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)


# In[6]:


train_y = df_train['label'].astype(int).values
valid_y = df_valid['label'].astype(int).values


# In[7]:


df_testa['label']=-1
df_data=pd.concat([df_train,df_valid,df_testa]).reset_index(drop=True)


# In[39]:


df_train.label.value_counts()


# # cv Statis

# In[38]:


def clean_csr(csr_trn, csr_sub, min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    sub_min = {x for x in np.where(csr_sub.getnnz(axis=0) >= min_df)[0]}
    mask= [x for x in trn_min if x in sub_min]
    return csr_trn[:, mask], csr_sub[:, mask]
def statis_feat_smooth(df,feature,label = 'label'):
    
    click_column = "{feature}_click".format(column=feature)
    count_column = "{feature}_count".format(column=feature)
    ctr_column = "{feature}_ctr".format(column=feature)

    agg_dict = {click_column: "sum", count_column: "count"}

    df = df.groupby(feature)[label].agg(agg_dict).reset_index()
    new_feat_name = feature + '_cvstat_cvr'
    I = df[count_column]
    C = df[click_column]
    df.loc[:,new_feat_name] = C/(I+5)

    df.loc[:,new_feat_name] = np.round(df.loc[:,new_feat_name].values,6)
    
    df_stas = df[[feature,click_column,count_column,new_feat_name]]
    return df_stas
def statis_feat_smooth_and_merge(df,df_val, feature,label = 'label'):
    df_stas = statis_feat_smooth(df,feature,label =label)
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)
    return df_val

class CVRCVStatistic(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,min_count=0):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for concat_feat in self.feat_list:
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_cvr'
            self.cvr_feat_list.append(new_feat_name)
            if min_count>0:
                df_cv[concat_feat] = self._remove_lowcase(df_cv[concat_feat],min_count)
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = statis_feat_smooth_and_merge(X_train,X_val,concat_feat,label=self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = statis_feat_smooth_and_merge(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
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

class CrossFeatCVRCVStatistic(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df,min_count=0):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2 in self.feat_list:
            concat_feat = feat1+'_'+feat2
            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_cvr'
            self.cvr_feat_list.append(new_feat_name)
            if min_count>0:
                df_cv[concat_feat] = self._remove_lowcase(df_cv[concat_feat],min_count)
            
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = statis_feat_smooth_and_merge(X_train,X_val,concat_feat,label=self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = statis_feat_smooth_and_merge(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del training
            del predict
            del df[concat_feat]
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df
    
    def triple_cvs(self, df,min_count=0):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2,feat3 in self.feat_list:
            concat_feat = feat1+'_'+feat2+'_'+feat3

            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)+'_f3-'+df[feat3].astype(str)
            
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_cvr'
            self.cvr_feat_list.append(new_feat_name)
            if min_count>0:
                df_cv[concat_feat] = self._remove_lowcase(df_cv[concat_feat],min_count)
            
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = statis_feat_smooth_and_merge(X_train,X_val,concat_feat,label=self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = statis_feat_smooth_and_merge(training, predict, concat_feat,label=self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
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
        
class HotCVStatistic(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for concat_feat in self.feat_list:
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_hot'
            self.cvr_feat_list.append(new_feat_name)
          
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = self._statis(X_train,X_val,concat_feat,self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = self._statis(training, predict, concat_feat,self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
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
      
class CrossHotCVStatistic(object):
    """
    K 折交叉统计特征
    """
    def __init__(self, feat_list, label,random_state=2018):
        self.feat_list = feat_list
        self.label = label
        self.cvr_feat_list=[]
        self.random_state = random_state
#         self.stratified_col = stratified_col
    def cvs(self, df):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2 in self.feat_list:
            concat_feat = feat1+'_'+feat2
            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_hot'
            self.cvr_feat_list.append(new_feat_name)
          
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = self._statis(X_train,X_val,concat_feat,self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = self._statis(training, predict, concat_feat,self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
            del df[concat_feat]
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='_index')
            print(concat_feat,'done!')

        del df['_index']
        return df

    def triple_cvs(self, df):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        df['_index'] = list(range(df.shape[0]))
        for feat1,feat2,feat3 in self.feat_list:
            concat_feat = feat1+'_'+feat2+'_'+feat3
            df[concat_feat] = 'f1-'+df[feat1].astype(str)+'_f2-'+df[feat2].astype(str)+'_f3-'+df[feat3].astype(str)
            df_cv = df[['_index',concat_feat, self.label]]
            new_feat_name = concat_feat + '_cvstat_hot'
            self.cvr_feat_list.append(new_feat_name)
          
            training = df_cv[df_cv.label!=-1]  
            training = training.reset_index(drop=True)
            predict = df_cv[df_cv.label==-1]
            del df_cv
            gc.collect()

            df_stas_feat = None

            sf = StratifiedKFold(5,shuffle=True,random_state=self.random_state)
            stratified_se = training.label.astype(str)

            cnt = 0
            for train_index, val_index in sf.split(training,stratified_se.values):
                print(concat_feat+' split order: ',cnt)
                cnt = cnt+1
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]
                X_val = self._statis(X_train,X_val,concat_feat,self.label)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)#columns:(index,f,f_cvr,label)

            X_pred = self._statis(training, predict, concat_feat,self.label)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)#columns:(index,f,f_cvr,label)

            del df_stas_feat[self.label]
            del df_stas_feat[concat_feat]
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


cvr_features = ['prefix', 'title', 'tag']
cvstor=CVRCVStatistic(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,True)


# In[40]:


cvr_features = [('prefix','title'), ('prefix','tag')]
cvstor=CrossFeatCVRCVStatistic(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data)


# In[41]:


features = ['prefix', 'title']
cvstor=HotCVStatistic(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data)


# In[42]:


cvr_features = [('prefix','title'), ('prefix','tag')]
cvstor=CrossHotCVStatistic(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data)


cvr_features = [('title','tag')]
cvstor=CrossFeatCVRCVStatistic(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,3)

cvr_features = [ ('title','tag')]
cvstor=CrossHotCVStatistic(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data)

print("saving cvr feature")
df_data.to_csv('cvr_cv_feature.csv',index=False)
# # Text jieba cut

# In[8]:


# str_mapper = str.maketrans("","",string.punctuation)
# with utils.timer('jieba cut'):
#     df_data['cut_prefix'] = df_data['prefix'].apply(lambda x:jieba.lcut(x.translate(str_mapper)))
#     df_data['cut_title'] = df_data['title'].apply(lambda x:jieba.lcut(x.translate(str_mapper)))
#     df_data['cut_query_prediction'] = df_data['query_prediction'].apply(lambda x:{k:[[float(v)],jieba.lcut(k.translate(str_mapper))] for k,v in eval(x).items()})


# # In[12]:


# df_data['cut_prefix']


# # In[19]:


# from operator import itemgetter
# query_predict = sorted(df_data['query_prediction'].items(),key=itemgetter(1), reverse=True)


# def func(x):
#     text = []
#     text.extend(x['cut_prefix'])
#     text.extend(x['cut_title'])
#     preds=  []
#     for k,v in x['cut_query_prediction'].items():
#         preds.extend(v[1] )
#     text.extend(preds)
#     return text
# with utils.timer('gen text'):
#     df_data['words']=df_data.apply(lambda x:func(x),axis=1)


# # In[46]:


# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     query_prediction = x['cut_query_prediction']
#     score = 0.0
#     for k,v in query_prediction.items():
#         pred_words = v[1]
#         pred_words = [word for word in pred_words if word in title_words]
#         if len(pred_words)>0:
#             score +=v[0][0]
#     return score
# with utils.timer('title_pred_score'):
#     df_data['title_pred_score']=df_data.apply(lambda x:func(x),axis=1)


# # In[47]:


# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     query_prediction = x['cut_query_prediction']
#     nwords = 0.0
#     new_words= []
#     pred_words = []
#     for k,v in query_prediction.items():
#         pred_words.extend(v[1])
#     new_words=[word for word in title_words if word not in pred_words]
#     return len(new_words)
# with utils.timer('title_unseen_nword'):
#     df_data['title_unseen_nword']=df_data.apply(lambda x:func(x),axis=1)


# # In[48]:


# def func(x):
#     query_prediction = x['cut_query_prediction']
#     freqs = []
#     for k,v in query_prediction.items():
#         freqs.extend(v[0])
#     std = np.std(freqs) if len(freqs)>0 else 0.0
#     return std
# with utils.timer('pred_freq_std'):
#     df_data['pred_freq_std']=df_data.apply(lambda x:func(x),axis=1)


# # In[49]:


# def func(x):
#     query_prediction = x['cut_query_prediction']
#     freqs = []
#     for k,v in query_prediction.items():
#         freqs.extend(v[0])
#     std = np.mean(freqs) if len(freqs)>0 else 0.0
#     return std
# with utils.timer('pred_freq_std'):
#     df_data['pred_freq_mean']=df_data.apply(lambda x:func(x),axis=1)

# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     return len(title_words)
# with utils.timer('nword_title_unseen_in_prefix'):
#     df_data['nword_title_unseen_in_prefix']=df_data.apply(lambda x:func(x),axis=1)


# # In[54]:


# def func(x):
#     query_prediction = x['cut_query_prediction']
#     freqs = []
#     for k,v in query_prediction.items():
#         freqs.extend(v[0])
#     _sum = np.sum(freqs) if len(freqs)>0 else 0.0
#     return _sum
# with utils.timer('pred_freq_sum'):
#     df_data['pred_freq_sum']=df_data.apply(lambda x:func(x),axis=1)


# # In[55]:


# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     query_prediction = x['cut_query_prediction']
#     score = 0.0
#     fit_nwords=[]
#     fit_keys = []
#     for k,v in query_prediction.items():
#         pred_words = v[1]
#         pred_words = [word for word in pred_words if word in title_words]
#         if len(pred_words)>0:
#             fit_nwords.append(len(pred_words))
#             fit_keys.append(k)
#     if len(fit_keys)>0:
#         k = fit_keys[np.argmax(fit_nwords)]
#         score = query_prediction[k][0][0]
#     return score
# with utils.timer('title_unseen_in_prefix_score_max'):
#     df_data['title_unseen_in_prefix_score_max']=df_data.apply(lambda x:func(x),axis=1)


# # In[56]:


# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     query_prediction = x['cut_query_prediction']
#     score = 0.0
    
#     scores = []
#     for k,v in query_prediction.items():
#         pred_words = v[1]
#         pred_words = [word for word in pred_words if word in title_words]
#         if len(pred_words)>0:
#             scores.append(v[0])
#     std = np.std(scores) if len(scores)>0 else 0.0
#     return std
# with utils.timer('title_unseen_in_prefix_score_std'):
#     df_data['title_unseen_in_prefix_score_std']=df_data.apply(lambda x:func(x),axis=1)


# # In[57]:


# def func(x):
#     words = []
#     prefix_words = x['cut_prefix']
#     title_words = x['cut_title']
#     title_words = [word for word in title_words if word not in prefix_words]
#     query_prediction = x['cut_query_prediction']
#     score = 0.0
    
#     scores = []
#     for k,v in query_prediction.items():
#         pred_words = v[1]
#         pred_words = [word for word in pred_words if word in title_words]
#         if len(pred_words)>0:
#             scores.append(v[0])
#     mean = np.mean(scores) if len(scores)>0 else 0.0
#     return mean
# with utils.timer('title_unseen_in_prefix_score_mean'):
#     df_data['title_unseen_in_prefix_score_mean']=df_data.apply(lambda x:func(x),axis=1)


# # In[58]:


# with utils.timer('prefix_nwords'):
#     df_data['prefix_nwords']=df_data['cut_prefix'].apply(lambda x: len(x))




# fillna


for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


print("cvr data shape:",df_data.shape)

#merge other feature
query_df = pd.read_csv('query_df_pq.csv')
prefix_df = pd.read_csv('prefix_df_pq.csv')
text_df = pd.read_csv('text_df_pq.csv')

drop_columns = ['prefix', 'title']
prefix_df = prefix_df.drop(columns=drop_columns)

drop_columns = ['prefix', 'query_prediction', 'tag', 'title', 'label']
text_df = text_df.drop(columns=drop_columns)

df_data = pd.concat([df_data, prefix_df, query_df, text_df], axis=1)

# In[51]:

#fillna

with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot',
    'prefix_w2v','max_similar','mean_similar','weight_similar']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)

# In[61]:


# df_valid.shape


# In[62]:


df_train = df_data.iloc[:df_train.shape[0]]
df_valid= df_data.iloc[df_train.shape[0]:df_train.shape[0]+df_valid.shape[0]]
df_testa = df_data.iloc[df_train.shape[0]+df_valid.shape[0]:]


# In[63]:




# In[64]:


#df_train.head()['title_unseen_in_prefix_score_max']


# In[ ]:


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


# In[76]:


# import zipfile
# a = zipfile.ZipFile('test.zip','w')
# a.write('mean-1.csv')
# a.close()


# In[ ]:



# In[74]:
features = ['prefix_cvstat_cvr','title_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot']
features+= ['title_pred_score','title_unseen_nword', 'pred_freq_std', 'pred_freq_mean']
features+=['title_tag_cvstat_cvr']
features+=['title_tag_cvstat_hot']
features+=['prefix_nwords','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std',
           'title_unseen_in_prefix_score_mean','nword_title_unseen_in_prefix' ,'pred_freq_sum']
features+=['max_similar', 'mean_similar', 'weight_similar'] # query feature
features+=['is_in_title', 'leven_distance','distance_rate', 'prefix_w2v']

#features+=['title_query_cosine_similar']
with utils.timer('Stack Feature'):
    train_x= sparse.hstack((cv_train,tag_train))
    valid_x = sparse.hstack((cv_valid,tag_valid))
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
        valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


# In[75]:


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=16
)
with utils.timer('Train LGB'):
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)


# In[116]:

# 找最佳阈值

# y_prob = clf.predict_proba(valid_x)
# y_prob = y_prob[:,1]
# for thr in np.linspace(0.3,0.7,20):
#     pred = (y_prob>thr).astype(np.uint32)
#     print(float("%0.4f"%thr),":valid f1_score: ",f1_score(valid_y, pred),sum(pred)/len(pred))


# In[117]:


# y_pred = clf.predict(testa_x)
thr = 0.4053
y_prob = clf.predict_proba(testa_x)
y_prob = y_prob[:,1]

y_pred = (y_prob>thr).astype(np.uint32)
save_dir = '../tmp/lgb_ml_bl_1023/submit/'
os.makedirs(save_dir,exist_ok=True)
with open(save_dir+'lgb_ml_bl_1023_'+str(thr)+'.csv','w') as fout:
    for pred in y_pred:
        fout.write(str(pred)+'\n')