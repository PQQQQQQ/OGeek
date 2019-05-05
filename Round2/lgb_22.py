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

import zipfile

warnings.filterwarnings('ignore')


prefix_df = pd.read_csv('../PQ/prefix_df_pq.csv')
ctr_df = pd.read_csv('../PQ/ctr_df_pq_1.csv')
query_df = pd.read_csv('../PQ/query_df_pq.csv')
text_df = pd.read_csv('../PQ/text_df_pq.csv')
stat_df = pd.read_csv('../PQ/stat_df_pq.csv')
nunique_df = pd.read_csv('../PQ/nunique_df_pq.csv')
# complete_df = pd.read_csv('complete_ctr_df_cv_all.csv')
com_prefix_df = pd.read_csv('../PQ/comp_prefix_df.csv')
# add_df = pd.read_csv('add_df_pq.csv')
# kmeans_df_1 = pd.read_csv('kmeans_prefix_w2v_2.csv')
# kmeans_df_2 = pd.read_csv('kmeans_title_w2v_2.csv')

drop_columns = ['prefix', 'title']
prefix_df = prefix_df.drop(columns=drop_columns)
drop_columns_p = ['complete_prefix', 'title']
com_prefix_df = com_prefix_df.drop(columns=drop_columns_p)

# drop_columns_c = ['cate_prefix', 'cate_title']
# ctr_df = ctr_df.drop(columns=drop_columns_c)

drop_columns_1 = ['prefix', 'query_prediction', 'tag', 'title', 'label']
text_df = text_df.drop(columns=drop_columns_1)
# 'prob_sum', 'prob_mean'
drop_columns_s = ['label', 'tag', 'complete_prefix', 'prefix_word_num', 'title_len', 'query_length']
stat_df = stat_df.drop(columns=drop_columns_s)


df = pd.concat([ctr_df, prefix_df, query_df, text_df, stat_df, nunique_df , com_prefix_df], axis=1)
del ctr_df, prefix_df, query_df, text_df, stat_df, nunique_df , com_prefix_df
gc.collect()

for col in df.columns:
    print(col,':',df[col].isnull().sum())

# for col in df.columns:
#     if col not in ['label']:
#         df[col]=df[col].fillna(df[col].mean())

# for col in df.columns:
#     print(col,':',df[col].isnull().sum())

drop_columns_2 = ['prefix', 'query_prediction', 'title', 'prefix_num', 'title_num', 'cut_prefix', 'cut_title', 'cut_query_prediction', 'words']
df = df.drop(columns=drop_columns_2)
df['tag']=le.fit_transform(df['tag'])

# df = pd.get_dummies(df, columns=['tag'])
for col in df.columns:
    df[col]=df[col].fillna(0)#df[col].mean()
for col in df.columns:
    print(col,':',df[col].isnull().sum())   
    
df['prob_max_ratio']=df['prob_max']/(df['prob_sum']+0.000001)
train_df_length = 2000000
validate_df_length = 50000


train_data = df[:train_df_length]
train_data["label"] = train_data["label"].apply(int)

validate_data = df[train_df_length:train_df_length + validate_df_length]
validate_data["label"] = validate_data["label"].apply(int)

test_data = df[train_df_length + validate_df_length:]
test_data = test_data.drop(columns=["label"])


# train_data_name = "train_pq.csv"
# validate_data_name = "validate_pq.csv"
# test_data_name = "test_pq.csv"

# train_df = pd.read_csv('train_pq.csv')
# validate_df = pd.read_csv('validate_pq.csv')
# test_df = pd.read_csv('test_pq.csv')

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
    
    y_prob_test.to_csv("prob_test.csv",index=False, header=False)
    
    y_prob_val.to_csv("prob_valid.csv",index=False, header=False)
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


for i in np.arange(0.35,0.5,0.001):
    label_valid=np.array([1 if y>=i else 0 for y in y_prob_valid])
    label_test=np.array([1 if y>=i else 0 for y in y_prob_test])
    print(i,f1_score(validate_data["label"].values, label_valid),np.mean(label_valid),np.mean(label_test))