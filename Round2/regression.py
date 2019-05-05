

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

features = ['prefix_cvstat_cvr','title_cvstat_cvr','tag_cvstat_cvr','prefix_title_cvstat_cvr','prefix_tag_cvstat_cvr','title_tag_cvstat_cvr']
features+= ['prefix_cvstat_hot','title_cvstat_hot','prefix_title_cvstat_hot','prefix_tag_cvstat_hot','title_tag_cvstat_hot']
# features+= ['prefix_self_hot','title_self_hot','prefix_title_self_hot','prefix_tag_self_hot','title_tag_self_hot']
features+= ['title_unseen_nword', 'pred_freq_std', 'pred_freq_mean','pred_freq_sum']

features+=['prefix_nwords','title_pred_score','title_unseen_in_prefix_score_max','title_unseen_in_prefix_score_std','title_unseen_in_prefix_score_mean',\
           'nword_title_unseen_in_prefix' ]

with utils.timer('Stack Feature'):
    reg_train_x= sparse.hstack((cv_train,tag_train)).tocsr()[train_keep_indices.tolist()]
    reg_valid_x = sparse.hstack((cv_valid,tag_valid)).tocsr()[valid_keep_indices.tolist()]
    testa_x = sparse.hstack((cv_testa,tag_testa))
    for feat in features:
        print(feat)
        reg_train_x = sparse.hstack((reg_train_x, df_train_reg[feat].values.reshape(-1,1)))
        reg_valid_x = sparse.hstack((reg_valid_x, df_valid_reg[feat].values.reshape(-1,1)))
        testa_x = sparse.hstack((testa_x, df_testa[feat].values.reshape(-1,1)))

reg_train_y = df_train_reg['score'].values
reg_valid_y = df_valid_reg['score'].values
_valid_y = valid_y[valid_keep_indices.tolist()]

clf = lgb.LGBMRegressor(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='regression',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=8
)
with utils.timer('Train LGB'):
    clf.fit(reg_train_x, reg_train_y, eval_set=[(valid_x, valid_y)], eval_metric=my_f1_score,early_stopping_rounds=100)