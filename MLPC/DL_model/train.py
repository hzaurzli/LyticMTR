import os
import tensorflow as tf
import keras
import numpy as np
np.random.seed(101)
from pathlib import Path



def catch(data, label):
    # preprocessing label and data
    l = len(data)
    chongfu = 0
    for i in range(l):
        ll = len(data)
        idx = []
        each = data[i]
        j = i + 1
        bo = False
        while j < ll:
            if (data[j] == each).all():
                label[i] += label[j]
                idx.append(j)
                bo = True
            j += 1
        t = [i] + idx
        if bo:
            chongfu += 1
        data = np.delete(data, idx, axis=0)
        label = np.delete(label, idx, axis=0)

        if i == len(data)-1:
            break
    print('total number of the same data: ', chongfu)

    return data, label



from model import *

def train_my(train, para, model_num, model_path):

    Path(model_path).mkdir(exist_ok=True)
    
    print('starting')

    # data get
    X_train, y_train = train[0], train[1]

    # data and label preprocessing
    y_train = keras.utils.to_categorical(y_train)
    X_train, y_train = catch(X_train, y_train)
    y_train[y_train > 1] = 1

    # disorganize
    index = np.arange(len(y_train))
    np.random.shuffle(index)
    X_train = X_train[index]
    y_train = y_train[index]

    # train
    length = X_train.shape[1]
    out_length = y_train.shape[1]

    for counter in range(1, model_num+1):
        # neural network model
        if model_path == 'CNN_base':           
            model = CNN_base(length, out_length, para)
        elif model_path == 'BiGRU_base':
            model = BiGRU_base(length, out_length, para)
        else:
            print('no model')
        
        model.fit(X_train, y_train, nb_epoch=50, batch_size=64, verbose=2)
        each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
        model.save(each_model)

       
import time
from test import test_my
def train_main(train, test, model_num, dir):

    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}

    train_my(train, para, model_num, dir)

    #test_my(test, para, model_num, dir)
