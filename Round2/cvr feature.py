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

