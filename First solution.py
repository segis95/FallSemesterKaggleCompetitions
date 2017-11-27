
# coding: utf-8

# In[2]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

get_ipython().run_line_magic('matplotlib', 'inline')


# We use xgboost as the principal alg

# In[5]:


import xgboost as xgb


# Preprocessing: 

# In[ ]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test["id"]

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train_y = train['target']
train_x = train.drop('target', axis = 1)

categorial = []
binary = []
continues = []

for f in train_x.columns:         
    # Defining the level
    if 'bin' in f:
        binary.append(f)
    elif 'cat' in f:
        categorial.append(f)
    elif train[f].dtype == float:
        continues.append(f)
    else:# train[f].dtype == int:
        categorial.append(f)
        
for c in categorial:
    temp = pd.concat([pd.Series(train_y), pd.Series(train_x[c])],axis = 1)
    freqs = temp.groupby(by = c).agg(["mean"])
    dic = freqs.to_dict()[('target', 'mean')]
    dic = defaultdict(lambda: 0.0, dic)
    L = [dic[x] for x in train_x[c]]
    try:
        K = [dic[x] for x in test[c]]
    except(KeyError):
        print(dic)
        print(c)
        break
    test[c] = K
    train_x[c] = L


# We prepare and train model using the xgboost alg

# In[15]:


dtrain = xgb.DMatrix(train_x.values, train_y.values)
dtest = xgb.DMatrix(test.values)


# In[16]:


param = {}
param['objective'] = 'binary:logistic'
param['eta'] = 0.02
param['silent'] = True
param['max_depth'] = 5
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['eval_metric'] = 'auc'


# In[17]:


evallist  = [(dtrain,'eval'), (dtrain,'train')]


# In[18]:


model=xgb.train(param, dtrain, 963, evallist, early_stopping_rounds=100, maximize=True, verbose_eval=9)


# We build a prediction based on our model

# In[19]:


pred = model.predict(dtest)


# Above we had extracted the id's(test_ids) from the test set(before erase them)

# Here we create a dataframe of needed format(corresponding to the Kaggle rules)

# In[34]:


res = pd.concat([test_ids, pd.Series(pred)],axis = 1)


# In[36]:


res.columns = ['id', 'target']


# In[40]:


res.to_csv('submit.csv',index = False)

