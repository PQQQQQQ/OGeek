import pandas as pd 
from collections import defaultdict
import pdb
import utils
import pickle as pkl
import os
class FeatureInfo:
    def __init__(self,name,empty_val,n_val):
        self.name = name
        # include empty
        self.empty_val = empty_val
        self.n_val = n_val
def Se_float2ffm(se):
    
    return se.apply(lambda x:str(float('%.8f'%x))+':1')
    
def Se_cate2ffm(se,enc_dict=None):
    if enc_dict is None:
        cates = pd.unique(se)
        enc_dict = defaultdict(int)
        for index,cate in enumerate(cates,1):
            enc_dict[cate]=index
    return se.apply(lambda x:'1.0:'+str(enc_dict[x])),enc_dict
def ret_none():
    return None
def convert2ffm(df_data,ignore_cols = ['label'],float_cols =[],cate_cols =[],enc_dicts=defaultdict(ret_none),convert_remain=False):
    df_data['id']=df_data.index
    converted_cols = ['id']
    ignore_cols.extend(['id','label'])
    ignore_cols = list(set(ignore_cols))
    # float
    for feat in float_cols:
        if pd.api.types.is_float_dtype(df_data[feat]):
            df_data[feat] = Se_float2ffm(df_data[feat])
        else:
            print(feat,'is forced to float for converting!')
            df_data[feat] = Se_float2ffm(df_data[feat])
    #
    for feat in cate_cols:
        if pd.api.types.is_string_dtype(df_data[feat]):
            df_data[feat],enc_dicts[feat] = Se_cate2ffm(df_data[feat],enc_dicts[feat])
        else:
            print(feat,'is forced to string for converting!')
            df_data[feat] = df_data[feat].apply(lambda x:str(x))
            df_data[feat],enc_dicts[feat] = Se_cate2ffm(df_data[feat],enc_dicts[feat])
    converted_cols.extend(float_cols)
    converted_cols.extend(cate_cols)
    if convert_remain:
        for feat in df_data.columns:
            if feat in ignore_cols:
                continue
            if feat in float_cols:
                continue
            if feat in cate_cols:
                continue
            if pd.api.types.is_float_dtype(df_data[feat]):
                df_data[feat] = Se_float2ffm(df_data[feat])
                converted_cols.append(feat)
            elif pd.api.types.is_string_dtype(df_data[feat]):
                converted_cols.append(feat)
                df_data[feat],enc_dicts[feat] = Se_cate2ffm(df_data[feat],enc_dicts[feat])
            else:
                print(feat,'is unconverted!')
    return df_data,enc_dicts,converted_cols

if __name__ == '__main__':
#     testa_file = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/PQ/test_pq.csv'
#     train_file = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/PQ/train_pq.csv'
#     valid_file = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/PQ/validate_pq.csv'
#     df_valid = pd.read_csv(valid_file)
#     print(df_valid['tag_click'].isnull().sum())
#     print(df_valid['tag_click'].dtype)
#     df_valid ,enc_dicts,converted_cols= convert2ffm(df_valid,float_cols=['tag_知道'],cate_cols=['tag_健康'])
#     print(converted_cols)
#     print(enc_dicts)
#     print(df_valid.head())
#     print(df_valid.head()['prefix_ctr'])
#     print(df_valid.head()['tag_知道'])
#     print(df_valid.head(50)['tag_健康'])
    data_file = '/home/admin/jupyter/Demo/DataSets/oppo_data_ronud2_20181107/PQ/cvr_cv_feature_v3.csv'
    df_data = pd.read_csv(data_file)
    print(df_data.columns)
    print(df_data.head())
    with utils.timer('fillna'):
        features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr', 'title_tag_cvstat_cvr']
        for col in features:
            df_data[col].fillna(df_data[col].mean(), inplace=True)
        features = ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
        for col in features:
            df_data[col].fillna(0.0, inplace=True)
       
    float_features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
    float_features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
#     features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']
#     features+= ['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
#                'nword_title_unseen_in_prefix' ]
    cate_features = ['tag']
     # check nan
    print('checking nan...')
    for col in float_features+cate_features:
        if df_data[col].isnull().sum()!=0:
            print(col,':',df_data[col].isnull().sum())
    # check features
    print('checking features...')
    for feat in float_features+cate_features:
        if feat not in df_data.columns:
            print(feat,' is not in data!')
    len_train = 2000000
    df_train = df_data.iloc[:len_train]
    df_valid= df_data[df_data['label']!=-1].iloc[len_train:]
    df_testa = df_data[df_data['label']==-1]
    with utils.timer('convert2ffm'):
        df_train ,enc_dicts,converted_cols= convert2ffm(df_train,float_cols=float_features,cate_cols=cate_features)
        df_valid ,enc_dicts,converted_cols= convert2ffm(df_valid,float_cols=float_features,cate_cols=cate_features,enc_dicts=enc_dicts)
        df_testa ,enc_dicts,converted_cols= convert2ffm(df_testa,float_cols=float_features,cate_cols=cate_features,enc_dicts=enc_dicts)
        save_dir = '../data/ffm_bl/'
        os.makedirs(save_dir,exist_ok=True)
        df_train[converted_cols+['label']].to_csv(save_dir+'train.csv',index=False)
        df_valid[converted_cols+['label']].to_csv(save_dir+'valid.csv',index=False)
        df_testa[converted_cols+['label']].to_csv(save_dir+'testa.csv',index=False)
        with open(save_dir+'enc_dicts.pkl','wb') as f:
            pkl.dump(enc_dicts,f)
        print(enc_dicts)
        with open(save_dir+'feat_infos.pkl','wb') as f:
            featureInfos = {}
            for feat in float_features:
                featureInfos[feat] = FeatureInfo(feat,0,2)
            for feat in cate_features:
                print(max(enc_dicts[feat].values()))
                featureInfos[feat] = FeatureInfo(feat,0,max(enc_dicts[feat].values())+1)
            print(featureInfos)
            pkl.dump(featureInfos,f)
            