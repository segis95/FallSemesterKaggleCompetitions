
# coding: utf-8

# In[2]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


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



#%% Feature selection

#def scale_and_pca(train_data, test_data):
#    print("Scaling...")
#    scaler = StandardScaler()
#    pca = PCA(n_components = 0.95, svd_solver = 'full', whiten = True)
#    
#    train_data = scaler.fit_transform(train_data)
#    test_data = scaler.fit_transform(test_data)
#    
#    print("Performing PCA...")
#    train_data = pca.fit_transform(train_data)
#    test_data = pca.transform(test_data)
#    print("New train size "+str(train_data.shape))
#    print("New test size "+str(test_data.shape))
#    return pd.DataFrame(train_data), pd.DataFrame(test_data)
#
#train_x, test = scale_and_pca(train_x, test)

#scaler = StandardScaler()
#column_names = train_x.columns
#train_x = pd.DataFrame(scaler.fit_transform(train_x), columns = column_names)
#test = pd.DataFrame(scaler.fit_transform(test), columns = column_names)

#th = .005
#
#selector = VarianceThreshold(threshold=th)
#selector.fit(train_x) # Fit to train without id and target variables
#
#f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements
#
#low_variance_variables = train_x.columns[f(selector.get_support())]
#
#print(str(len(low_variance_variables))+" variables had variance < "+str(th)+" , dropped.")
#
#train_x = train_x.drop(low_variance_variables.get_values(),axis=1)
#test = test.drop(low_variance_variables.get_values(),axis=1)





# In[15]:

# We prepare and train model using the xgboost alg

#dtrain = xgb.DMatrix(train_x.values, train_y.values)
#dtest = xgb.DMatrix(test.values)


# In[16]:


#param = {}
#param['booster'] = 'gbtree'
#param['objective'] = 'binary:logistic'
#param['tree_method'] ='hist'
#param['eta'] = 0.02
#param['silent'] = True
#param['max_depth'] = 5
#param['subsample'] = 0.8
#param['colsample_bytree'] = 0.8
#param['eval_metric'] = 'auc'

model = xgb.XGBClassifier(    
                        n_estimators=400,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                        eval_metric='auc',
                        silent= True
                     )

# In[17]:


#evallist  = [(dtrain,'eval'), (dtrain,'train')]


# In[18]:

#print("Boosted trees. Starting...")

#model=xgb.train(param, dtrain, 963, evallist, early_stopping_rounds=100, maximize=True, verbose_eval=50)


# We build a prediction based on our model

# In[19]:


y_valid_pred = 0*train_y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

for i, (train_index, test_index) in enumerate(kf.split(train_x, train_y)):
    
    # Create data for this fold
    y_train, y_valid = train_y.iloc[train_index].copy(), train_y.iloc[test_index]
    X_train, X_valid = train_x.iloc[train_index,:].copy(), train_x.iloc[test_index,:].copy()
    X_test = test.copy()
    print( "\nFold ", i)
    
    # Enocode data
    for c in categorial:
        temp = pd.concat([pd.Series(y_train), pd.Series(X_train[c])],axis = 1)
        freqs = temp.groupby(by = c).agg(["mean"])
        dic = freqs.to_dict()[('target', 'mean')]
        dic = defaultdict(lambda: 0.0, dic)
        dictrain = [dic[x] for x in X_train[c]]
        dicvalid = [dic[x] for x in X_valid[c]]
        dictest = [dic[x] for x in X_test[c]]
        X_train[c] = dictrain
        X_valid[c] = dicvalid
        X_test[c] = dictest
        
    # Run model for this fold
    fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    
    del X_test, X_train, X_valid, y_train
    
pred /= K  # Average test set predictions  
#pred = model.predict(dtest)


# Above we had extracted the id's(test_ids) from the test set(before erase them)

# Here we create a dataframe of needed format(corresponding to the Kaggle rules)



# In[34]:


res = pd.concat([test_ids, pd.Series(pred)],axis = 1)


# In[36]:


res.columns = ['id', 'target']


# In[40]:


res.to_csv('submit2.csv',index = False)

print("End...")

#%% Gini
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0.
    gini = 0.
    delta = 0.
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1. - 2. * gini / (ntrue * (n - ntrue))
    return gini

eval_gini(train_y, y_valid_pred)