
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import utils
import json
import re


# In[2]:


query_df = pd.read_csv('feature/query_df_b.csv')


# In[4]:


query_df.isnull().sum()


# In[5]:


nunique_df = pd.read_csv('feature/nunique_df_b.csv')


# In[6]:


nunique_df.columns.values


# In[7]:


nunique_df = nunique_df.drop(columns=['prefix_title', 'prefix_tag', 'title_tag'])


# In[8]:


nunique_df.to_csv('feature/nunique_df_b.csv',index=False)


# In[9]:


ctr_df  = pd.read_csv('feature/ctr_df_b.csv')


# In[10]:


ctr_df.columns.values


# In[11]:


stat_df = pd.read_csv('feature/stat_df_b.csv')


# In[12]:


stat_df.columns.values


# In[ ]:


columns = train_data.columns
# print('train shape:',train_data.shape)
# print('train feature:',columns)
start_time=time.time()
remove_columns = ["label"]
features_columns = [column for column in columns if column not in remove_columns]

train_data_1 = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
train_features = train_data_1[features_columns]
train_labels = train_data_1["label"]


# In[7]:


train_pq = pd.read_csv('train_pq.csv')


# In[8]:


train_pq.shape


# In[9]:


train_pq.columns.values


# In[4]:


#youwenti cate_prefix cate_title
train_pq.columns.values


# In[6]:


feature = ['label', 'prefix_click', 'prefix_count', 'prefix_ctr',
       'title_click', 'title_count', 'title_ctr', 'tag_click',
       'tag_count', 'tag_ctr', 'prefix_title_click', 'prefix_title_count',
       'prefix_title_ctr', 'prefix_tag_click', 'prefix_tag_count',
       'prefix_tag_ctr', 'title_tag_click', 'title_tag_count',
       'title_tag_ctr', 'cate_prefix_click', 'cate_prefix_count',
       'cate_prefix_ctr', 'cate_title_click', 'cate_title_count',
       'cate_title_ctr', 'cate_prefix_cate_title_click',
       'cate_prefix_cate_title_count', 'cate_prefix_cate_title_ctr',
       'cate_prefix_tag_click', 'cate_prefix_tag_count',
       'cate_prefix_tag_ctr', 'cate_title_tag_click',
       'cate_title_tag_count', 'cate_title_tag_ctr',
       'prefix_title_tag_click', 'prefix_title_tag_count',
       'prefix_title_tag_ctr', 'prefix_num_title_num_tag_click',
       'prefix_num_title_num_tag_count', 'prefix_num_title_num_tag_ctr',
       'is_in_title', 'leven_distance', 'distance_rate', 'prefix_w2v',
       'max_similar', 'mean_similar', 'weight_similar',
       'title_pred_score', 'title_unseen_nword', 'pred_freq_std',
       'pred_freq_mean', 'nword_title_unseen_in_prefix', 'pred_freq_sum',
       'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'max_query_ratio', 'title_word_num', 'small_query_num', 'prob_max',
       'complete_prefix_click', 'complete_prefix_count',
       'complete_prefix_ctr', 'complete_prefix_title_click',
       'complete_prefix_title_count', 'complete_prefix_title_ctr',
       'complete_prefix_tag_click', 'complete_prefix_tag_count',
       'complete_prefix_tag_ctr', 'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c', 'tag_健康', 'tag_商品', 'tag_应用',
       'tag_影视', 'tag_快应用', 'tag_旅游', 'tag_景点', 'tag_歌手', 'tag_汽车',
       'tag_游戏', 'tag_火车', 'tag_百科', 'tag_知道', 'tag_经验', 'tag_网站',
       'tag_网页', 'tag_航班', 'tag_菜谱', 'tag_酒店', 'tag_阅读', 'tag_音乐']


# In[8]:


len(feature)


# In[5]:


train_pq.columns.values


# In[5]:


feature = ['label', 'prefix', 'query_prediction', 'tag', 'title',
       'prefix_num', 'title_num', 'prefix_click', 'prefix_count',
       'prefix_ctr', 'title_click', 'title_count', 'title_ctr',
       'tag_click', 'tag_count', 'tag_ctr', 'prefix_title_click',
       'prefix_title_count', 'prefix_title_ctr', 'prefix_tag_click',
       'prefix_tag_count', 'prefix_tag_ctr', 'title_tag_click',
       'title_tag_count', 'title_tag_ctr', 'cate_prefix_click',
       'cate_prefix_count', 'cate_prefix_ctr', 'cate_title_click',
       'cate_title_count', 'cate_title_ctr',
       'cate_prefix_cate_title_click', 'cate_prefix_cate_title_count',
       'cate_prefix_cate_title_ctr', 'cate_prefix_tag_click',
       'cate_prefix_tag_count', 'cate_prefix_tag_ctr',
       'cate_title_tag_click', 'cate_title_tag_count',
       'cate_title_tag_ctr', 'prefix_title_tag_click',
       'prefix_title_tag_count', 'prefix_title_tag_ctr',
       'prefix_num_title_num_tag_click', 'prefix_num_title_num_tag_count',
       'prefix_num_title_num_tag_ctr', 'is_in_title', 'leven_distance',
       'distance_rate', 'prefix_w2v', 'max_similar', 'mean_similar',
       'weight_similar', 'title_pred_score', 'title_unseen_nword',
       'pred_freq_std', 'pred_freq_mean', 'nword_title_unseen_in_prefix',
       'pred_freq_sum', 'title_unseen_in_prefix_score_max',
       'title_unseen_in_prefix_score_std',
       'title_unseen_in_prefix_score_mean', 'prefix_nwords',
       'max_query_ratio', 'title_word_num', 'small_query_num', 'prob_max',
       'is_in_title_c', 'leven_distance_c', 'distance_rate_c',
       'prefix_w2v_c', 'complete_prefix_click', 'complete_prefix_count',
       'complete_prefix_ctr', 'complete_prefix_title_click',
       'complete_prefix_title_count', 'complete_prefix_title_ctr',
       'complete_prefix_tag_click', 'complete_prefix_tag_count',
       'complete_prefix_tag_ctr']


# In[6]:


len(feature)


# In[17]:


train_pq = train_pq.drop(columns=['complete_prefix_title_tag', 'prefix_title', 'prefix_tag',
       'complete_prefix_title', 'complete_prefix_tag', 'title_tag',
       'complete_prefix_title_tag.1'])


# In[19]:


train_pq.to_csv('nunique_df_pq_2.csv',index=False)


# In[9]:


nunique = pd.read_csv('nunique_df_pq.csv')


# In[10]:


nunique.columns.values


# In[11]:


add_df_pq = pd.read_csv('add_df_pq.csv')


# In[6]:


cvr = pd.read_csv('cvr_cv_df_3.csv')


# In[3]:


stat_df = pd.read_csv('stat_df_pq.csv')


# In[5]:


stat_df.columns.values


# In[6]:


nunique_df = pd.read_csv('nunique_df_pq.csv')


# In[7]:


nunique_df.columns.values


# In[8]:


query_df = pd.read_csv('query_df_pq.csv')


# In[9]:


query_df.columns.values


# In[10]:


text_df = pd.read_csv('text_df_pq.csv')


# In[11]:


text_df.columns.values


# In[12]:


complete_prefix_df = pd.read_csv('comp_prefix_df.csv')


# In[13]:


complete_prefix_df.columns.values


# In[8]:


cvr.columns.values


# In[6]:


ctr = pd.read_csv('ctr_df_pq_1.csv')


# In[12]:


ctr.columns.values


# In[ ]:





# In[14]:


prefix_df = pd.read_csv('prefix_df_pq.csv')


# In[15]:


prefix_df.columns.values


# In[3]:


vali = utils.read_txt('../data_vali.txt')
train = utils.read_txt('../data_train.txt')
test = utils.read_txt('../data_test.txt',is_label=False)


# In[7]:


train_length = train.shape[0]
vali_length = vali.shape[0]
test_length = test.shape[0]


# In[4]:


data=pd.concat([train,vali,test]).reset_index(drop=True)


# In[5]:


add_ctr = pd.read_csv('add_ctr_df_pq.csv')


# In[10]:


add_ctr['label'] = data['label']


# In[11]:


train_data = add_ctr[:train_length]
validate_data = add_ctr[train_length:train_length+vali_length]


# In[20]:


validate_data.head()


# In[12]:


columns = train_data.columns

remove_columns = ["label"]
features_columns = [column for column in columns if column not in remove_columns]

# train_data = pd.concat([train_data, validate_data], axis=0, ignore_index=True)
train_features = train_data[features_columns]
train_labels = train_data["label"]

# validate_data_length = validate_data.shape[0]
validate_features = validate_data[features_columns]
validate_labels = validate_data["label"]


# In[13]:


x = pd.concat([train_features,train_labels],axis=1)
y = pd.concat([validate_features,validate_labels],axis=1)


# In[22]:


from featexp import get_trend_stats
from featexp import get_univariate_plots
get_univariate_plots(data = x, target_col='label',                      data_test = y, features_list=['num_pred_title_ctr'],bins=10)


# In[15]:


from featexp import get_trend_stats
from featexp import get_univariate_plots
stats = get_trend_stats(data=x, target_col='label',                      data_test=y)


# In[17]:


complete_prefix = pd.read_csv('comp_prefix_df.csv')


# In[21]:


complete_prefix.drop(columns=['title', 'is_in_title_c', 'leven_distance_c',
       'distance_rate_c', 'prefix_w2v_c'])


# In[23]:


data = pd.concat([data,complete_prefix],axis=1)


# In[24]:


data.head()


# In[10]:


def move_useless_char(s):
    # 提出无效字符
    return re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+??！，。？?、~@#￥%……&*（）]+", "", s)
def query_prediction_text(query_prediction):
    if (query_prediction == "{}") | (query_prediction == "") | pd.isna(query_prediction) | (query_prediction == "nan"):
        return ["PAD"]
    json_data = json.loads(query_prediction)
    result = sorted(json_data.items(), key=lambda d: d[1], reverse=True)
    texts = [move_useless_char(item[0]) for item in result]
    return texts


# In[13]:


data['query_word'] = data['query_prediction'].apply(lambda x: query_prediction_text(x))


# In[14]:


data['query_word'].head()


# In[25]:


def deal_data_query_sub_word_info(x):
    # 对每个 query_word 删除 prefix
    try:
        rst = [re.sub(x['complete_prefix'], "", _x) for _x in x['query_word']] if len(x['query_word']) > 0 else ['NAN']
    except:
        rst = [_x for _x in x['query_word']]
    return rst
data['query_sub_word'] = data[['prefix', 'query_word']].apply(lambda x: deal_data_query_sub_word_info(x), axis=1)


# In[16]:


data['query_sub_word'].head()


# In[26]:


data['query_sub_word'].head()


# In[ ]:


def deal_eig_value(similarity_matrix):
    # similarity_matrix: 对称矩阵
    similarity_matrix = np.array(similarity_matrix)
    similarity_matrix = similarity_matrix + similarity_matrix.T
    similarity_matrix[np.eye(similarity_matrix.shape[0]) == 1] = 1
    eig_value = np.linalg.eig(similarity_matrix)[0]
    eig_value = [float(x) for x in eig_value]
    eig_value = sorted(eig_value, reverse=True) + [0 for _ in range(10 - len(eig_value))]
    return eig_value


def deal_query_word_mutual_text_eig_vector(sub_word):
    # 计算query_word 中词组包含关系信息主向量
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [1-(len(sw)-len(_sw))/max([len(sw), len(_sw)]) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix)  # 计算特征向量特征值
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


# In[ ]:


def deal_query_word_levenshtein_ratio_eig_vector(sub_word):
    # 计算query_word的 levenshetein 相似度
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [Levenshtein.ratio(_sw, sw) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix) # 计算特征向量
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value

def deal_query_word_levenshtein_distance_eig_vector(sub_word):
    # 计算query_word的 levenshetein 相似度
    sub_word = [x for x in sub_word if x != ""]
    if len(sub_word) > 0:
        similarity_matrix = []
        for _sw in sub_word:
            similarity = [Levenshtein.distance(_sw, sw) if _sw in sw else 0 for sw in sub_word ]
            similarity_matrix.append(similarity)
        eig_value = deal_eig_value(similarity_matrix) # 计算特征向量
    else:
        eig_value = [0 for _ in range(10)]
    return eig_value


# In[ ]:


# data['query_sub_word'].head()
eig_values = data['query_word'].apply(lambda x: deal_query_word_mutual_text_eig_vector(x))
pred_list=[]
data['mutual_text_eig_value_sum'] = []
for i in range(10):
    data['mutual_text_eig_value_'+str(i)] = eig_values.apply(lambda x: x[i])
    
    data['mutual_text_eig_value_sum'] += data['mutual_text_eig_value_'+str(i)] 
#     pred_list.append(data['mutual_text_eig_value_'+str(i)])
#     data['mutual_text_eig_value_sum'] = np.sum(pred_list)
#     data['mutual_text_eig_value_mean'] = np.mean(pred_list)
#     data['mutual_text_eig_value_std'] = np.std(pred_list)
#data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')


# In[ ]:


eig_values = data['query_word'].apply(lambda x: deal_query_word_levenshtein_ratio_eig_vector(x))
pred_list=[]
data['levenshtein_ratio_eig_value_sum']  = []
for i in range(10):
    data['levenshtein_ratio_eig_value_'+str(i)] = eig_values.apply(lambda x: x[i])
    
    data['levenshtein_ratio_eig_value_sum'] += data['levenshtein_ratio_eig_value_'+str(i)] 
#data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')


# In[ ]:


eig_values = data['query_word'].apply(lambda x: deal_query_word_levenshtein_distance_eig_vector(x))
pred_list=[]
data['levenshtein_distance_eig_value_sum'] = []
for i in range(10):
    data['levenshtein_distance_eig_value_' + str(i)] = eig_values.apply(lambda x: x[i])
    data['levenshtein_distance_eig_value_sum'] += data['levenshtein_distance_eig_value_'+str(i)]
    #     pred_list.append(data['levenshtein_distance_eig_value_'+str(i)])
#     data['levenshtein_distance_eig_value_sum'] = np.sum(pred_list)
#     data['levenshtein_distance_eig_value_mean'] = np.mean(pred_list)
#     data['levenshtein_distance_eig_value_std'] = np.std(pred_list)
#data = data.merge(temp_data.drop(['query_word', 'query_sub_word'], axis=1), on='prefix', how='left')


# In[ ]:




