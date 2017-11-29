
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sc
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# In[ ]:


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

from sklearn.metrics import roc_auc_score

class roc_callback(keras.callbacks.Callback):
    
    def __init__(self,training_data,validation_data, display):
        self.display = display
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.seen = 0


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        #print(epoch)
        if int(epoch) % self.display == 0:
            y_pred_val = self.model.predict_proba(self.x_val)
            roc_val = roc_auc_score(self.y_val, y_pred_val)
            print('\gini: %s ' % (2 * roc_val - 1))
            return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import xgboost as xgb
class FeatureBinarizatorAndScaler:
    """ This class needed for scaling and binarization features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])


    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((
                len(data[feature]), 1))), axis=1)
        print(binarizedAndScaledFeatures.shape)
        return binarizedAndScaledFeatures


def preproc(X_train):
    # Adding new features and deleting features with low importance
    multreg = X_train['ps_reg_01'] * X_train['ps_reg_03'] * X_train['ps_reg_02']
    ps_car_reg = X_train['ps_car_13'] * X_train['ps_reg_03'] * X_train['ps_car_13']
    X_train = X_train.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',
                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',
                            'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',
                            'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_10_cat', 'ps_ind_10_bin',
                            'ps_ind_13_bin', 'ps_ind_12_bin'], axis=1)
    X_train['mult'] = multreg
    X_train['ps_car'] = ps_car_reg
    X_train['ps_ind'] = X_train['ps_ind_03'] * X_train['ps_ind_15']
    return X_train


# In[ ]:


train_x = pd.read_csv('train.csv')
train_y = train_x['target']
train_x = train_x.drop(['id', 'target'], axis=1)
test_x = pd.read_csv('test.csv')
test_x = test_x.drop(['id'], axis=1)
train_x = preproc(train_x)
test_x = preproc(test_x)


# In[ ]:


train_y1 = abs(-1+train_y)
y_train = pd.concat([pd.Series(train_y), pd.Series(train_y1)], axis=1)
y_train = y_train.as_matrix()


# In[ ]:



#hyperparameters
input_dimension = 36
learning_rate = 0.0025
momentum = 0.85
hidden_initializer = random_uniform(seed=2)
dropout_rate = 0.3


# create model
model = Sequential()
model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(100, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(100, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))
model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))

sgd = SGD(lr=learning_rate, momentum=momentum)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(train_x.values, y_train, epochs=250, batch_size=128,verbose = 1,callbacks=[roc_callback(training_data=(train_x.values, y_train),validation_data=(train_x.values, y_train),display = 10)] )

predictions = model.predict_proba(X_test)
#print(gini_normalized(y_test,predictions))

