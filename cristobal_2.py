import time
t0 = time.time()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pk
from collections import defaultdict

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


 



def sergei_pre():

    train = pd.read_csv('data/train.csv')

    test = pd.read_csv('data/test.csv')

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

    return train_x, train_y, test, test_ids

#####################################################################
#####################################################################

train_x, train_y, test, test_ids = sergei_pre()

x_train, x_test, y_train, y_test = train_test_split(train_x,
                                                    train_y,
                                                    test_size = val_split)

print('time until preprocessing is done: ', round(time.time() - t0, 3))

#print('data shape: ', train_data.shape)

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
    
    regularization = 10**(-4*reg_exp - 3)
    learning_rate = 10**(-2*lr_exp - 3)
    layers = [Dense(50,
                input_shape = (x_train.shape[1],),
                activation = 'relu',
                kernel_regularizer = reg.l2(regularization))]

    #add 5 layers wi
    for i in range(5):
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

if not os.path.isfile('models/model_2.h5'):

    weights1 = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train.values),
                                                     y_train.values)

    weights1 = {i: weights1[i] for i in range(len(weights1))}

    layers = [Dense(50,
                    input_shape = (x_train.shape[1],),
                    activation = 'relu',
                    kernel_regularizer = reg.l2(regularization))]


    for i in range(5):
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
    history = model.fit(x_train.values, y_train.values,
                        epochs = epochs,
                        batch_size = batch,
                        verbose = 0,
                        validation_data = (x_test.values, y_test.values),
                        class_weight = weights1)

    print('time until model has finished training: ', round(time.time() - t0, 3))
    print('true error: ',
          np.mean(history.history['val_acc'][-5:]))
    print('time until error has evaluated: ', round(time.time() - t0, 3))

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('models/model_2.h5')

else:
    model = load_model('models/model_2.h5')

    weights1 = class_weight.compute_class_weight('balanced',
                                                     np.unique(y_train.values),
                                                     y_train.values)

    weights1 = {i: weights1[i] for i in range(len(weights1))}

    history = model.fit(x_train.values, y_train.values,
                        epochs = epochs,
                        batch_size = batch,
                        verbose = 0,
                        validation_data = (x_test.values, y_test.values),
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

    model.save('models/model_2.h5')

pred = model.predict(test.values)[:, 0]

res = pd.concat([test_ids, pd.Series(pred)], axis = 1)
res.columns = ['id', 'target']

num = len(os.listdir('submits')) + 1

res.to_csv('submits/submission_'+str(num), index = False)

##################################### save experiment into an excel ######################

#df_parameters = 

#################################### with sklearn ###########################

##clf = ens.AdaBoostClassifier()
##clf.fit(train_data, train_labels)
##print(clf.score(train_data, train_labels))


