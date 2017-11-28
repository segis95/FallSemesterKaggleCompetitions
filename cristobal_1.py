import time
t0 = time.time()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pk

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, GaussianDropout
from keras.initializers import RandomNormal, RandomUniform, TruncatedNormal
from keras.optimizers import Adam, Adadelta
import keras.optimizers as kopt
import keras.regularizers as reg

import sklearn.neural_network as nn
import sklearn.ensemble as ens
import sklearn.linear_model as lm
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


#################################### parameters

best_params = {'reg_exp_1': 0.6517968154887782,
'reg_exp': 0.7371698374615214,
'reg_exp_2': 0.4371162594318422}

#[0.51329336826700067, 0.77284390591589447]

pca_per = 0.95
learn_rate = 10**(-(2)*best_params['reg_exp_1'] - 2)
#learn_rate = 0.002
epochs = 30
batch = int(20000*best_params['reg_exp_2'])
val_split = 0.33
regularization = 10**(-(2)*best_params['reg_exp'] - 3)


####################################
if os.path.isfile('data/filtered_data'):
    with open('data/filtered_data', 'rb') as handle:
        data_tuple = pk.load(handle)
    train_data, train_labels = data_tuple[0], data_tuple[1]
    df_test = pd.read_pickle('data/test')

elif os.path.isfile('data/sample_submission')\
   and os.path.isfile('data/train')\
   and os.path.isfile('data/test'):

    df_sample = pd.read_pickle('data/sample_submission')
    
    df_train = pd.read_pickle('data/train')

    df_test = pd.read_pickle('data/test')

else:
    #read data files from CSVs
    df_sample = pd.read_csv('data/sample_submission.csv', index_col = 0)

    df_train = pd.read_csv('data/train.csv', index_col = 0)
    df_test = pd.read_csv('data/test.csv', index_col = 0)

    #save dataframes as pickle files
    df_sample.to_pickle('data/sample_submission')

    df_train.to_pickle('data/train')
    df_test.to_pickle('data/test')

print('time until packages and dataframe are loaded: ', round(time.time() - t0, 3))

######################## set binary and categorical columns to dummies

def add_dummies(df_train):
    categorical_cols = [col for col in df_train.columns
     if col.endswith('_cat') or col.endswith('_bin')]

    continuous_cols = [col for col in df_train.columns
                       if not col in categorical_cols]


    #dummies for categorical variables
    df_train = pd.get_dummies(df_train,
                              prefix = categorical_cols,
                              columns = categorical_cols)


    #dummies for continuous variables
    d_cols = [col for col in continuous_cols
              if np.any(df_train[col] == -1)]

    df_dummies = pd.get_dummies(df_train[d_cols] == -1,
                                columns = d_cols,
                                drop_first = True)

    df_train = pd.concat([df_train, df_dummies], axis = 1)
    df_train[df_train == -1] = 0

    return df_train

#df_train = add_dummies(df_train)

######################## balance the affirmative cases and the negative cases

##df_affirmative = df_train[df_train.target == 1]
##len_pos = len(df_affirmative)
##len_neg = len(df_train) - len_pos
##
##df_train_balanced = pd.concat([df_train]
##                              + [df_affirmative]*(int(len_neg/len_pos) - 1))


########################


def get_data_and_labels(df_train, labels = True):
    if labels:
        train_labels = df_train.iloc[:, 0].values
        train_data = df_train.iloc[:, 1:].values

        return train_data, train_labels
    else:
        return df_train.values

#train_data, train_labels = get_data_and_labels(df_train)


##train_labels_balanced = df_train_balanced.iloc[:, 0].values
##train_data_balanced = df_train_balanced.iloc[:, 1:].values


#test_labels = df_test.iloc[:, 0].values
#test_data = df_test.values

######################## scaling and PCA #########################


def scale_and_pca(train_data):
    print(train_data.shape)

    scaler = StandardScaler()
    pca = PCA(n_components = pca_per, svd_solver = 'full', whiten = True)

    train_data = scaler.fit_transform(train_data)
    train_data = pca.fit_transform(train_data)

    print(train_data.shape)
    return train_data, scaler, pca #needs adjustment, added scaler and pca

#train_data = scale_and_pca(train_data)



#####################################################################
#####################################################################


def preprocess_data(df, target = True):
    df = add_dummies(df)
    if target:
        data, labels = get_data_and_labels(df)
        data, scaler, pca= scale_and_pca(data)
        return data, scaler, pca, labels
    
    else:
        data = get_data_and_labels(df, False)
        data = scale_and_pca(data)
        return data, scaler, pca

    
if not os.path.isfile('data/filtered_data'):
    train_data, train_labels = preprocess_data(df_train)


#####################################################################
#####################################################################

x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_labels,
                                                    test_size = val_split)

print('time until preprocessing is done: ', round(time.time() - t0, 3))

print('data shape: ', train_data.shape)

def data():
    with open('data/filtered_data', 'rb') as handle:
        data_tuple = pk.load(handle)
    train_data, train_labels = data_tuple[0], data_tuple[1]
    
    x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_labels,
                                                    test_size = 0.33)
    return x_train, y_train, x_test, y_test



def create_model(x_train, y_train, x_test, y_test):

    t0 = time.time()
    
    weights1 = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)

    weights1 = {i: weights1[i] for i in range(len(weights1))}

    reg_exp = {{uniform(0, 1)}}
    lr_exp = {{uniform(0, 1)}}
    batch_proportion = {{uniform(0, 1)}}
    
    regularization = 10**(-(2)*reg_exp - 3)
    learning_rate = 10**(-(2)*lr_exp - 2)
    layers = [Dense(50,
                input_shape = (x_train.shape[1],),
                activation = 'relu',
                kernel_regularizer = reg.l2(regularization))]

    #add 5 layers wi
    for i in range(20):
        layers.append(Dense(units = 50,
                            activation = 'relu',
                            kernel_regularizer = reg.l2(regularization)))

    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))
              
    model = Sequential(layers)

    opt = kopt.Nadam(lr = learning_rate)

    model.compile(optimizer = opt,
                  loss = 'binary_crossentropy',
                  metrics = ['acc'])
    t1 = time.time()

    history = model.fit(x_train, y_train,
                        epochs = 130,
                        batch_size = int(20000*batch_proportion),
                        verbose = 0,
                        validation_data = (x_test, y_test),
                        class_weight = weights1)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #score, acc = model.evaluate(x_test, y_test, verbose=0)

    acc = np.mean(history.history['val_acc'][-5:])

    adj_acc = acc - 10*np.var(history.history['val_acc'][-20:])**2 
    
    print('Test accuracy:', acc)
    print('variance of last 20 terms: ',
          np.var(history.history['val_acc'][-20:]))
    print('Overall error (acc - 10*var^2): ', adj_acc)
    print('time elapsed: ', round((time.time() - t0)/60, 2), ' min')
    return {'loss': -adj_acc, 'status': STATUS_OK, 'model': model}

##############################################################################

##best_run, best_model, space = optim.minimize(model=create_model,
##                                          data=data,
##                                          algo=tpe.suggest,
##                                          max_evals=6,
##                                          trials=Trials(),
##                                          eval_space=True,
##                                          return_space=True)
##print("Evalutation of best performing model:")
##print(best_model.evaluate(x_test, y_test, verbose = 0))
##print("Best performing model chosen hyper-parameters:")
##print(best_run)


###############################################################################

if not os.path.isfile('current_model.h5'):

    weights1 = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_labels),
                                                     train_labels)

    weights1 = {i: weights1[i] for i in range(len(weights1))}

    layers = [Dense(50,
                    input_shape = (train_data.shape[1],),
                    activation = 'relu',
                    kernel_regularizer = reg.l2(regularization))]


    for i in range(20):
    ##    layers.append(Dense(50,
    ##                  kernel_initializer = init,
    ##                  bias_initializer = init))
        layers.append(Dense(units = 50,
                            activation = 'relu',
                            kernel_regularizer = reg.l2(regularization)))
        #layers.append(Activation('relu'))
        #layers.append(GaussianDropout(0.1))

    layers.append(Dense(1))
    layers.append(Activation('sigmoid'))
              
    model = Sequential(layers)
    opt = kopt.Nadam(learn_rate)

    model.compile(optimizer = opt,
                  loss = 'binary_crossentropy',
                  metrics = ['acc', 'binary_accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs = epochs,
                        batch_size = batch,
                        verbose = 0,
                        validation_split = val_split,
                        class_weight = weights1)

    print('time until model has finished training: ', round(time.time() - t0, 3))


    print('true error: ',
          np.mean(history.history['val_acc'][-5:]))
    ##print('balanced error: ', 
    ##      model.evaluate(train_data_balanced, train_labels_balanced, verbose = 0))

    print('time until error has evaluated: ', round(time.time() - t0, 3))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('current_model.h5')

else:
    model = load_model('current_model.h5')

    weights1 = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_labels),
                                                     train_labels)

    weights1 = {i: weights1[i] for i in range(len(weights1))}

    history = model.fit(train_data, train_labels,
                        epochs = epochs,
                        batch_size = batch,
                        verbose = 0,
                        validation_split = val_split,
                        class_weight = weights1)

    print('time until model has finished training: ', round(time.time() - t0, 3))


    print('true error: ',
          np.mean(history.history['val_acc'][-5:]))
    ##print('balanced error: ', 
    ##      model.evaluate(train_data_balanced, train_labels_balanced, verbose = 0))

    print('time until error has been evaluated: ', round(time.time() - t0, 3))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('current_model.h5')

##pred = model.predict(preprocess_data(df_test, False))
##
##sample = pd.read_csv('data/sample_submission.csv')
##test_ids = sample['id']
##
##res = pd.concat([test_ids, pd.series(pred)], axis = 1)
##res.columns = ['id', 'target']
##
##num = len(os.listdir(submits) + 1)
##
##res.to_csv('submission_'+str(num), index = False)

##################################### save experiment into an excel ######################

#df_parameters = 

#################################### with sklearn ###########################

##clf = ens.AdaBoostClassifier()
##clf.fit(train_data, train_labels)
##print(clf.score(train_data, train_labels))


