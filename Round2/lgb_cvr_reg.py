%load_ext autoreload
%autoreload 2
%matplotlib inline
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

def my_f1_score(labels, preds):
    return 'f1_score', f1_score(labels,preds.round()), True

root_dir = '../../DataSets/oppo_data_ronud2_20181107/'

df_valid = utils.read_txt(root_dir+'data_vali.txt')
df_train = utils.read_txt(root_dir+'data_train.txt')
df_testa = utils.read_txt(root_dir+'data_test.txt',is_label=False)

df_train['label'] = df_train['label'].astype(int)
df_valid['label'] = df_valid['label'].astype(int)
train_y = df_train['label'].astype(int).values
valid_y = df_valid['label'].astype(int).values

df_testa['label']=-1
df_train['dataset_type']=-1
df_valid['dataset_type']=-2
df_testa['dataset_type']=-3
df_data=pd.concat([df_train,df_valid,df_testa]).reset_index(drop=True)

df_data['query_prediction'] = df_data['query_prediction'].apply(lambda x: x if x!='' else '{}')

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

cvr_features = ['prefix', 'title', 'tag']
cvstor=CVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10,is_test_use_val=False)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossFeatCVRCVStatistic2(cvr_features,'label')
with utils.timer('CVR'):
    df_data = cvstor.cvs(df_data,0,kfold=10,is_test_use_val=False)

features = ['prefix', 'title']
cvstor=HotCVStatistic2(features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10,is_test_use_val=False)
cvr_features = [('prefix','title'), ('prefix','tag'),('title','tag')]
cvstor=CrossHotCVStatistic2(cvr_features,'label')
with utils.timer('Hot'):
    df_data = cvstor.cvs(df_data,kfold=10,is_test_use_val=False)

# saving cvr feature
print("saving cvr feature")
df_data.to_csv('cvr_1124.csv',index=False)

#reading our other features
print("reading other features")
query_df = pd.read_csv('query_df_pq.csv')
prefix_df = pd.read_csv('prefix_df_pq.csv')
text_df = pd.read_csv('text_df_pq.csv')
ctr_df = pd.read_csv('ctr_df_pq_1.csv')
complete_prefix_df = pd.read_csv('complete_prefix_df.csv')
nunique_df = pd.read_csv('nunique_df_pq.csv')
stat_df = pd.read_csv('stat_df_pq.csv')


df_data = pd.concat([df_data,query_df,prefix_df,text_df,ctr_df,complete_prefix_df,nunique_df,stat_df],axis=1)

#df_data['words'] = df_data['texts'].apply(lambda x: x.split(' '))

for col in df_data.columns:
    print(col,':',df_data[col].isnull().sum())


with utils.timer('fillna'):
    features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr']
    for col in features:
        df_data[col].fillna(df_data[col].mean(), inplace=True)
    features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
    for col in features:
        df_data[col].fillna(0.0, inplace=True)

df_train = df_data[df_data['dataset_type']==-1].reset_index(drop=True)
df_valid = df_data[df_data['dataset_type']==-2].reset_index(drop=True)
df_testa = df_data[df_data['dataset_type']==-3].reset_index(drop=True)

with utils.timer('CountVectorizer'):
    cvor = CountVectorizer(analyzer=lambda x: x,binary=True)
    cv_train = cvor.fit_transform(df_train['words']) 
    cv_valid = cvor.transform(df_valid['words']) 
    cv_testa = cvor.transform(df_testa['words']) 

lb_enc = LabelEncoder()
enc = OneHotEncoder()
with utils.timer('Encoder'):
    tag_train = lb_enc.fit_transform(df_train['tag'])
    tag_valid = lb_enc.transform(df_valid['tag'])
    tag_testa = lb_enc.transform(df_testa['tag'])

    tag_train = enc.fit_transform(tag_train.reshape(-1, 1))
    tag_valid = enc.transform(tag_valid.reshape(-1, 1))
    tag_testa = enc.transform(tag_testa.reshape(-1, 1))

def clean_csr_2(csr_trn,  min_df):
    trn_min = np.where(csr_trn.getnnz(axis=0) >= min_df)[0]
    return csr_trn[:, trn_min], trn_min
with utils.timer('clean'):
    cv_train,mask = clean_csr_2(cv_train,2)
    cv_valid = cv_valid[:,mask]
    cv_testa = cv_testa[:,mask]


# features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
# features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
# # features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
# features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

# features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
#            'nword_title_unseen_in_prefix' ]

# with utils.timer('Stack Feature'):
#     train_x= sparse.hstack((cv_train,tag_train))
#     valid_x = sparse.hstack((cv_valid,tag_valid))
#     testa_x = sparse.hstack((cv_testa,tag_testa))
#     for feat in features:
#         print(feat)
#         train_x= sparse.hstack((train_x, df_train[feat].values.reshape(-1,1)))
#         valid_x = sparse.hstack((valid_x, df_valid[feat].values.reshape(-1,1)))
#         testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))


#regression
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

reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]

features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    reg_testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        reg_testa_x = sparse.hstack((reg_testa_x, df_testa[feat].values.reshape(-1,1)))

clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(reg_valid_x, reg_valid_y)], early_stopping_rounds=100)


y_prob = clf.predict(valid_x)
y_prob_test = clf.predict(testa_x)
for thr in np.linspace(0.3,0.7,20):
    pred = (y_prob>thr).astype(np.uint32)
    pred_test = (y_prob_test>thr).astype(np.uint32)
    f1 = f1_score(valid_y, pred)
    mean_pred=sum(pred)/len(pred)
    mean_pred_test = sum(pred_test)/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))      