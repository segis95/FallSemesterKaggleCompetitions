
# coding: utf-8

# In[2]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold

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

#%% Out of fold prediction

param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param['tree_method'] ='hist'
param['eta'] = 0.1
param['silent'] = True
param['max_depth'] = 5
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['eval_metric'] = 'auc'


def probability_to_rank(prediction, scaler=1.):
    pred_df=pd.DataFrame(columns=['probability'])
    pred_df['probability']=prediction
    pred_df['rank']=pred_df['probability'].rank()/len(prediction)*scaler
    return pred_df['rank'].values


def cv_xgb(params, x_train, y_train, x_test, kf, cat_cols=[], verbose=True, 
                       verbose_eval=50, num_boost_round=4000, use_rank=True):

    # initialise the size of out-of-fold train an test prediction
    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        x_test_kf=x_test.copy()

        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)
        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)
        d_test = xgb.DMatrix(x_test_kf)

        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,
                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,
                        early_stopping_rounds=50)

        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)
        
        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred+=probability_to_rank(bst.predict(d_test))
        else:
            train_pred[val_index] += val_pred
            test_pred+=bst.predict(d_test)

    test_pred /= kf.n_splits

    return train_pred,test_pred
    

#Level 1
kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)

xgb_train_pred, xgb_test_pred = cv_xgb(param, train_x, train_y, test, kf, use_rank=False, verbose_eval=False)

xgb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_train_pred)
xgb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_test_pred)


#Level 2



# In[15]:


dtrain = xgb.DMatrix(train_x.values, train_y.values)
dtest = xgb.DMatrix(test.values)


# In[16]:




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
