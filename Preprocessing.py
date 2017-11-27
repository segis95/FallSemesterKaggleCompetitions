
# coding: utf-8

# In[58]:

import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#get_ipython().magic(u'matplotlib inline')



# In[60]:

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train_y = train['target']
train_x = train.drop('target', axis = 1)

categorial = []
binary = []
continues = []
integer = []

for f in train_x.columns:         
    # Defining the level
    if 'bin' in f:
        binary.append(f)
    elif 'cat' in f:
        categorial.append(f)
    elif train[f].dtype == float:
        continues.append(f)
    else:# train[f].dtype == int:
        integer.append(f)

# Target based encoding
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


# In[21]:




# In[51]:




# In[34]:




# In[59]:




# In[49]:




# In[50]:

np.isnan(train_x).sum(axis = 0)


# In[3]:
unique, counts = np.unique(x, return_counts=True)


# In[ ]:



