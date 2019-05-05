def get_tf(texts):
    tf = defaultdict(int)
    for doc in texts:
        for token in doc:
            tf[token] += 1
    return tf
def filter_tf(tf,min_tf):
    return dict((term, freq) for term, freq in tf.items()
                           if freq >= min_tf)

df_data['cut_prefix'] = df_data['cleaned_prefix'].apply(lambda x: jieba.lcut(x))

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