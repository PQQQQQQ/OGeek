
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import utils
import zipfile
from gensim.models.keyedvectors import KeyedVectors

import time
import json
import re
import jieba
import Levenshtein
import logging
import warnings
import pickle

from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn import metrics


# In[2]:


val_data = utils.read_txt('../data_vali.txt')
train_data = utils.read_txt('../data_train.txt')
test_data = utils.read_txt('../data_testb.txt',is_label=False)


# In[3]:


train_data['label'] = train_data['label'].astype(int)
val_data['label'] = val_data['label'].astype(int)


# In[4]:


train_data = pd.concat([train_data,val_data]).reset_index(drop=True)


# In[20]:


train_data.shape


# In[21]:


test_data['prefix'].isin(train_data['prefix']).sum()


# In[5]:


test_data_1 = test_data[:120000]


# In[23]:


test_data_1['prefix'].isin(train_data['prefix']).sum()


# In[9]:


pred_merge = pd.read_csv('pred_merge_1202.csv',header=None)


# In[12]:


pred_merge


# In[ ]:





# In[11]:


pred_merge.columns=['pred']


# In[14]:


pred_2fold = pd.read_csv('predict0.776107-12-02-21-34.csv',header=None)


# In[15]:


pred_2fold.columns= ['pred']


# In[16]:


pred_2fold['pred'].corr(pred_merge['pred'])


# In[17]:


test_data['pred1'] = pred_merge['pred']


# In[18]:


test_data['pred2'] = pred_2fold['pred']


# In[19]:


test_data


# In[16]:


# ctr_df = pd.read_csv('feature/ctr_df_b.csv')


# In[20]:


temp1 = train_data.groupby(['prefix','title','tag'],as_index=False)['label'].agg({'click1':'sum','count1':'count','ctr1':'mean'})


# In[21]:


a = pd.merge(test_data,temp1,on=['prefix','title','tag'],how='left')


# In[22]:


test_data['ctr'] = a['ctr1']


# In[27]:


test_data['pred3'] = stat_result['pred']


# In[28]:


test_data


# In[36]:


# test_data_1['ctr'] = a[:120000]['ctr1']


# In[29]:


test_data_1 = test_data[:120000]


# In[31]:


test_data_1['pred1'].mean()


# In[32]:


test_data_1['pred2'].mean()


# In[33]:


test_data_1['pred3'].mean()


# In[34]:


test_data_2 = test_data[120000:]


# In[35]:


test_data_2['pred1'].mean()


# In[36]:


test_data_2['pred2'].mean()


# In[37]:


test_data_2['pred3'].mean()


# In[38]:


test_data_1['pred1'] = (test_data_1['ctr']>0.40).astype(np.uint32)


# In[44]:


test_data_1[(test_data_1.ctr<0.5) & (test_data_1.ctr>0.4)][['pred','pred1','ctr']]


# In[24]:


stat_result = pd.read_csv('result.csv',header=None)


# In[26]:


stat_result.columns=['pred']


# In[8]:


test_data_1


# In[ ]:





# In[ ]:


test_data_2 = test_data[120000:]


# In[ ]:


test_data_2.mean()


# In[ ]:


stat_result_1 = stat_result_1[:120000]
stat_result_2 = stat_result_2[120000:]


# # 1202提交

# In[ ]:





# In[3]:


prob_cla = pd.read_csv('prob_test0.776760-12-01-22-49.csv',header=None)


# In[4]:


prob_cla.columns=['pred']


# In[7]:


prob_cla.head()


# In[5]:


prob_reg = pd.read_csv('/home/admin/jupyter/Demo/yao/tmp/lgb_ml_all_reg_1202/submit/lgb_ml_all_reg_12020.4.csv.prob',header=None)


# In[6]:


prob_reg.columns=['pred']


# In[9]:


prob_reg.head()


# In[10]:


pred_merge = pd.DataFrame()
pred_merge['pred'] = 0.7*prob_cla['pred']+0.3*prob_reg['pred']


# In[12]:


pred_merge = (pred_merge>0.40).astype(np.uint32)


# In[14]:


pred_merge.mean()


# In[22]:


pred_merge.shape


# In[15]:


pred_merge.to_csv('pred_merge_1202.csv',header=None,index=False)


# In[16]:


a = zipfile.ZipFile('test_1202.zip','w')
a.write("pred_merge_1202.csv")
a.close()


# In[17]:


pred_cla = (prob_cla>0.40).astype(np.uint32)


# In[18]:


pred_cla.mean()


# In[19]:


pred_reg = (prob_reg>0.40).astype(np.uint32)


# In[20]:


pred_reg.mean()


# In[24]:


pred_cla['pred'].corr(pred_merge['pred'])


# In[25]:


pred_diff = pd.DataFrame()
pred_diff['pred']=pred_cla['pred']- pred_reg['pred']


# In[29]:


pred_reg.nunique()


# In[30]:


pred_cla.nunique()


# In[34]:


pred_diff['pred'].value_counts()


# # 1130 提交

# In[16]:


prob_reg = pd.read_csv('prob_test0.774903-12-01-10-05.csv',header=None)


# In[3]:


prob_reg.columns=['pred']


# In[4]:


pred_reg = (prob_reg>0.40).astype(np.uint32)


# In[10]:


pred_reg.head()


# In[15]:


pred_reg


# In[7]:


pred_reg.to_csv('pred_1202.csv',header=None,index=False)


# In[8]:


a = zipfile.ZipFile('test_1202.zip','w')
a.write("pred_1202.csv")
a.close()


# In[5]:


pred_reg.mean()


# In[26]:


pred_reg['pred'].corr(pred_cla['pred'])


# In[11]:


prob_cla = pd.read_csv('/home/admin/jupyter/Demo/yao/tmp/lgb_ml_all_reg_1201/submit/lgb_ml_all_reg_12010.4.csv',header=None)


# In[12]:


prob_cla.columns = ['pred'] 


# In[13]:


prob_cla


# In[25]:


pred_cla = (prob_cla>0.405).astype(np.uint32)


# In[17]:


pred_cla.mean()


# In[15]:


prob_merge = pd.DataFrame()


# In[16]:


prob_merge['pred'] = 0.3*prob_cla['pred'] + 0.7*prob_reg['pred']


# In[24]:


pred_merge = (prob_merge>0.40).astype(np.uint32)


# In[18]:


pred_merge.to_csv('pred_merge_1201_0.40.csv',index=False,header=False)


# In[22]:


pred_merge.mean()


# In[ ]:


pred_merge


# In[20]:


a = zipfile.ZipFile('test_merge_1201.zip','w')
a.write("pred_merge_1201_0.40.csv")
a.close()


# In[46]:


pred_merge_1129 = pd.read_csv('pred_merge_1129.csv',header=None)


# In[47]:


pred_merge_1129.columns=['pred']


# In[48]:


pred_merge_1130 = pd.read_csv('pred_merge_1130_0.405.csv',header=None)


# In[49]:


pred_merge_1130.columns=['pred']


# In[50]:


pred_merge_1129['pred'].corr(pred_merge_1130['pred'])


# # 1129提交

# In[38]:


prob_yuehan = pd.read_csv('~/jupyter/Demo/yao/tmp/lgb_ml_all_reg_1129/submit/lgb_ml_all_reg_11290.4.csv.prob',header=None)


# In[39]:


predict_yuehan = pd.read_csv('/home/admin/jupyter/Demo/yao/tmp/lgb_ml_all_reg_1129/submit/lgb_ml_all_reg_11290.4.csv',header=None)


# In[40]:


predict_yuehan.columns=['pred']


# In[12]:


predict_yuehan.mean()


# In[41]:


prob_ty = pd.read_csv('prob_test0.774767-11-29-10-39.csv',header=None)


# In[14]:


prob_ty.mean()


# In[42]:


pred_ty = (prob_ty>0.4).astype(np.uint32)


# In[43]:


pred_ty.columns=['pred']


# In[45]:


predict_yuehan['pred'].corr(pred_ty['pred'])


# In[20]:


prob_ty_vali = pd.read_csv('prob_valid0.774767-11-29-10-39.csv',header=None)


# In[15]:


predicit_7799 = pd.read_csv('predict0.779945-11-28-11-47.csv',header=None)


# In[28]:


prob_merge = pd.DataFrame()


# In[30]:


prob_yuehan.columns=['pred']


# In[31]:


prob_ty.columns=['pred']


# In[34]:


prob_ty.head()


# In[35]:


prob_yuehan.head()


# In[36]:


prob_merge['pred'] = 0.7*prob_ty['pred'] + 0.3*prob_yuehan['pred']


# In[48]:


pred_merge = (prob_merge>0.405).astype(np.uint32)


# In[49]:


pred_merge.mean()


# In[51]:


pred_merge.shape


# In[52]:


pred_merge.head(10)


# In[53]:


pred_merge.to_csv('pred_merge_1129.csv',index=False,header=False)


# In[54]:


import zipfile


# In[56]:


a = zipfile.ZipFile('test_merge_1129.zip','w')
a.write("pred_merge_1129.csv")
a.close()


# In[25]:


for thr in np.linspace(0.35,0.55,40):
    pred = (prob_ty_vali>thr).astype(np.uint32)
    pred_test = (prob_ty>thr).astype(np.uint32)
    f1 = f1_score(validate_labels, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[ ]:





# In[16]:


predicit_7799.columns=['pred']


# In[27]:


pred_ty['pred'].corr(predicit_7799['pred'])


# In[8]:


# predict_yuehan = pd.read_csv('/home/admin/jupyter/Demo/yao/tmp/lgb_ml_all_reg_1129/submit/lgb_ml_all_reg_11290.4263.csv.prob',header=None)


# In[9]:


# predict_yuehan


# In[2]:


# predict_7301 = pd.read_csv('predict11-22-16-05.csv',header=None)
# predict_cv = pd.read_csv('predict_cv0.211-22-16-40.csv',header=None)
# predict_7301.columns=['pred']
# predict_cv.columns=['pred']


# In[4]:


predict_7301['pred'].corr(predict_cv['pred'])


# In[5]:


predict_1122 = pd.read_csv('xtyfile/submit/predict_1122.csv',header=None)


# In[6]:


predict_1122.columns=['pred']


# In[7]:


predict_7301['pred'].corr(predict_1122['pred'])


# In[12]:


predict_8021 = pd.read_csv('predict0.802102-11-19-22-59.csv',header=None)


# In[13]:


predict_8021.columns=['pred']


# In[14]:


predict_7301['pred'].corr(predict_8021['pred'])


# In[15]:


predict_cv['pred'].corr(predict_8021['pred'])


# In[2]:


predict_xzy = pd.read_csv('../xzy/predict0.778775-11-22-16-26.csv',header=None)


# In[ ]:


predict_xzy.columns=['pred']


# In[22]:


predict_7301['pred'].corr(predict_xzy['pred'])


# In[ ]:





# In[19]:


predict_xzy.mean()


# In[5]:


predict_xzy = pd.read_csv('../xzy/predict0.779761-11-23-10-40.csv',header=None)


# In[ ]:


predict_xzy.mean()


# In[7]:


predict_7301['pred'].corr(predict_xzy['pred'])


# In[24]:


predict_reg = pd.read_csv('predict_reg_0211-23-21-34.csv',header=None)


# In[25]:


predict_reg.columns=['pred']


# In[26]:


predict_7301['pred'].corr(predict_reg['pred'])


# In[27]:


predict_reg.mean()


# In[6]:


pred_clas = pd.read_csv('predict_classify_cv0.211-23-21-29.csv',header=None)


# In[7]:


pred_clas.columns=['pred']


# In[8]:


pred_clas.mean()


# In[9]:


pred_clas['pred'].corr(predict_reg['pred'])


# In[9]:


pred_10fold = pd.read_csv('../xzy/predict0.784955-11-24-10-06.csv',header=None)


# In[10]:


pred_10fold.shape


# In[12]:


pred_10fold.columns=['pred']


# In[13]:


predict_7301['pred'].corr(pred_10fold['pred'])


# In[16]:


df_prob1 = pd.read_csv('../../../yao/output/allfeat1/DNN/bl_adj/valid_res.csv_prob_4',header=None).rename(columns={0:'probs_4'})
df_sub1 = pd.read_csv('../../../yao/output/allfeat1/DNN/bl_adj/test_res.csv_prob_4',header=None).rename(columns={0:'prob'})


# In[17]:


df_prob1


# In[18]:


df_sub1


# In[19]:


y_pred = (df_sub1>0.4).astype(np.uint32)


# In[22]:


y_pred.columns = ['pred']


# In[28]:


predict_7301['pred'].corr(predict_reg['pred'])


# In[31]:


pred_merge = pd.DataFrame()
pred_merge['pred'] = 0.5*predict_7301['pred'] + 0.5*predict_reg['pred']


# In[38]:


pred_merge.mean()


# In[92]:


predict_1125 = pd.read_csv('predict1125.csv',header=None)


# In[93]:


predict_1125.columns = ['pred']


# In[94]:


predict_7301['pred'].corr(predict_1125['pred'])


# In[6]:


predict_1125.mean()


# In[90]:


pred_reg = pd.read_csv('prob_reg_11-25-21-05.csv',header=None)


# In[91]:


pred_reg.mean()


# In[45]:


vali_prob = pred_reg[:50000]


# In[46]:


test_prob = pred_reg[50000:]


# In[7]:


train_data = pd.read_csv('train_pq.csv')
validate_data = pd.read_csv('validate_pq.csv')
test_data = pd.read_csv('test_pq.csv')


# In[10]:


validate_labels = validate_data['label'].apply(int)


# In[13]:


mean_pred=sum(pred)/len(pred)


# In[17]:


len(pred)


# In[34]:


for thr in np.linspace(0.35,0.55,40):
    pred = (vali_prob>thr).astype(np.uint32)
    pred_test = (test_prob>thr).astype(np.uint32)
    f1 = f1_score(validate_labels, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[39]:


pred_test = (test_prob>0.4115).astype(np.uint32)


# In[40]:


pred_test  = pred_test.reset_index(drop = True)


# In[41]:


pred_test.columns=['pred']


# In[42]:


predict_7301['pred'].corr(pred_test['pred'])


# In[43]:


pred_test.mean()


# In[44]:


prob_p = pd.read_csv('prob_11252215p.csv',header=None)


# In[55]:


prob_p_t = pd.read_csv('prob_11252215p_t.csv',header=None)


# In[54]:


prob_p


# In[96]:


vali_prob = prob_p_t[:50000]
test_prob = prob_p_t[50000:]


# In[57]:


test_prob.shape


# In[97]:


for thr in np.linspace(0.35,0.55,40):
    pred = (vali_prob>thr).astype(np.uint32)
    pred_test = (test_prob>thr).astype(np.uint32)
    f1 = f1_score(validate_labels, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[98]:


pred_test = (test_prob>0.4526  ).astype(np.uint32)


# In[99]:


pred_test.columns = ['pred']


# In[78]:


pred_test.shape


# In[72]:





# In[71]:


predict_7301.shape


# In[101]:


pred_test  = pred_test.reset_index(drop = True)


# In[80]:


predict_7301['pred'].corr(pred_test['pred'])


# In[ ]:





# In[102]:


predict_1125['pred'].corr(pred_test['pred'])


# In[81]:


prob_cv_reg = pd.read_csv('prob_reg_11-25-22-24.csv',header=None)


# In[83]:


prob_cv_reg.columns=['pred']


# In[84]:


vali_prob = prob_cv_reg[:50000]
test_prob = prob_cv_reg[50000:]


# In[85]:


for thr in np.linspace(0.35,0.55,40):
    pred = (vali_prob>thr).astype(np.uint32)
    pred_test = (test_prob>thr).astype(np.uint32)
    f1 = f1_score(validate_labels, pred)
    mean_pred=pred.sum()/len(pred)
    mean_pred_test = pred_test.sum()/len(pred_test)
    print(float("%0.4f"%thr),":valid f1_score: ",float("%0.4f"%f1),float("%0.4f"%mean_pred),float("%0.4f"%mean_pred_test))


# In[87]:


pred_test = (test_prob>0.4115 ).astype(np.uint32)


# In[88]:


pred_test  = pred_test.reset_index(drop = True)


# In[95]:


predict_1125['pred'].corr(pred_test['pred'])


# In[103]:


prob_cla = pd.read_csv('prob_clas11-25-22-53.csv',header=None)


# In[156]:


prob_cla.mean()


# In[104]:


prob_cla.shape


# In[105]:


pred_reg = pd.read_csv('prob_reg_11-25-21-05.csv',header=None)


# In[106]:


pred_reg.shape


# In[107]:


prob_reg = pred_reg[50000:].reset_index(drop = True)


# In[110]:


prob_reg.columns = ['pred']


# In[111]:


prob_cla.columns = ['pred']


# In[112]:


prob_reg['pred'].corr(prob_cla['pred'])


# In[118]:





# In[ ]:





# In[125]:





# In[132]:


preb_reg = (prob_reg>0.40).astype(np.uint32)


# In[133]:


preb_reg.mean()


# In[134]:


preb_cla = (prob_cla>0.40).astype(np.uint32)


# In[135]:


preb_cla.mean()


# In[136]:


prob_merge = pd.DataFrame()
prob_merge['pred'] = 0.5*prob_reg['pred'] + 0.5*prob_cla['pred']


# In[147]:


preb_merge = (prob_merge>0.405).astype(np.uint32)


# In[148]:


preb_merge.mean()


# In[149]:


preb_merge['pred'].corr(predict_7301['pred'])


# In[152]:


preb_merge.to_csv('preb_merge.csv',index=False,header=False)


# In[153]:


import zipfile
a = zipfile.ZipFile('test_merge_cla_reg_1125.zip','w')
a.write("preb_merge.csv")
a.close()


# In[154]:


prob_cla_1 = pd.read_csv('prob_clas11-26-10-54.csv',header=None)


# In[155]:


prob_cla_1.mean()


# In[ ]:





# In[8]:


predict_1 = pd.read_csv('createfeature/predict0.780170-11-27-11-49.csv',header=None)


# In[ ]:





# In[11]:


predict_1.columns = ['pred']


# In[12]:


predict_2.columns = ['pred']


# In[14]:


predict_1['pred'].corr(predict_2['pred'])


# In[15]:


predict_2['pred'].corr(predict_xzy['pred'])


# In[10]:


predict_2 = pd.read_csv('createfeature/predict0.781648-11-27-11-38.csv',header=None)


# In[16]:


predict_3 = pd.read_csv('predict0.784162-11-28-10-10.csv',header=None)


# In[20]:


predict_3.columns = ['pred']


# In[17]:


predict_3.mean()


# In[23]:


predict_2['pred'].corr(predict_3['pred'])


# In[24]:


predict_4 = pd.read_csv('createfeature/predict0.781481-11-27-23-32.csv',header=None)


# In[25]:


predict_4.columns = ['pred']


# In[28]:


predict_3['pred'].corr(predict_4['pred'])


# In[29]:


predict_5 = pd.read_csv('predict0.782633-11-28-10-54.csv',header=None)


# In[30]:


predict_5.columns = ['pred']


# In[32]:


predict_5['pred'].corr(predict_2['pred'])


# In[33]:


predict_3['pred'].corr(predict_2['pred'])


# In[ ]:




