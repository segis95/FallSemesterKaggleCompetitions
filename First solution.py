
# coding: utf-8

# In[2]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#get_ipython().run_line_magic('matplotlib', 'inline')


# We use xgboost as the principal alg

# In[5]:


import xgboost as xgb


# Preprocessing: 

# In[ ]:

print("Loading files...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
test_ids = test["id"]

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

print("Filling missing values...")
mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)
mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
train_x['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()
train_x['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()
train_x['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()
train_x['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()

print("Target-based encoding categorical variables...")
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

#%% Feature selection
    
def scale_and_pca(train_data, test_data):
    print("Scaling...")
    scaler = StandardScaler()
    pca = PCA(n_components = 0.95, svd_solver = 'full', whiten = True)
    
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.fit_transform(test_data)
    
    print("Performing PCA...")
    train_data = pca.fit_transform(train_data)
    test_data = pca.transform(test_data)
    print("New train size "+str(train_data.shape))
    print("New test size "+str(test_data.shape))
    return pd.DataFrame(train_data), pd.DataFrame(test_data)

train_x, test = scale_and_pca(train_x, test)

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

print("Boosted trees. Starting...")

model=xgb.train(param, dtrain, 963, evallist, early_stopping_rounds=100, maximize=True, verbose_eval=50)


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

print("End...")
