import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.models import model_from_json
from evaluation import scores, evaluate
import pickle
from keras.models import load_model
from keras.backend import expand_dims
import numpy as np
import pandas as pd


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


def predict(X_test, y_test, para, weights, jsonFiles, h5_model, dir):
    
    adam = Adam(lr=para['learning_rate']) # adam optimizer
    # 1.loading weight and structure (model)
    h5_model_path = '/home/user/Desktop/yyf_data/MLBP/MLBP/model.h5'
    load_my_model = load_model(h5_model_path, custom_objects={'keras.backend': keras.backend, 'expand_dims': expand_dims}) # 需要传一个custom_objects参数,将自定义的层添加进去
    print("Prediction is in progress")

    # 2.predict
    score = load_my_model.predict(X_test)
    print(score)
    score_laber = score
    "========================================"
    for i in range(len(score_laber)):
        max_i = max(score_laber[i])
        for j in range(len(score_laber[i])):
            if score_laber[i][j] < max_i:
                score_laber[i][j] = 0
            else:
                score_laber[i][j] = 1
    
    functions = []      
    for e in score_laber:
      laber = []  
      for i in range(len(e)):
        if e[i] == 1:
          laber.append('1')
        else:
          laber.append('0')
      functions.append(laber)

    output_file = os.path.join('/home/user/Desktop/yyf_data/MLBP/MLBP/', 'result.txt')
    with open(output_file, 'w') as f:
        for i in functions:
            f.write('\t'.join(i) + '\n')

    
    "========================================"
    print("========================================")
    print(score_laber)
    print("========================================")
    print(y_test)

    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_laber, y_test)
    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')


def test_my(test, para, dir):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp

    # weight and json
    weights = []
    jsonFiles = []
    h5_model = []
    weights.append('model.hdf5')
    jsonFiles.append('model.json')
    h5_model.append('model.h5')
        
    # step2:predict
    predict(test[0], test[1], para, weights, jsonFiles, h5_model, dir)
    
if __name__ == '__main__':
  f = open('/home/user/Desktop/yyf_data/laber/random_sample/final_input.txt')
  g = open('/home/user/Desktop/yyf_data/laber/random_sample/laber.txt')
  
  max_seq_len = 500
  feats = []
  for i in f:
    feat_property = i.strip().split('\t')[1:6]
    feat_other = i.strip().split('\t')[6:]
    feat_property_padding = [0] * int(max_seq_len - len(feat_property))
    feat_property.extend(feat_property_padding)
    feat_property.extend(feat_other)
    feat_tmp = list(map(float,feat_property))
    feats.append(feat_tmp)
    
  laber_tmp = []
  for j in g:
    laber = j.strip().split('\t')[0]
    num = int(j.strip().split('\t')[1])
    laber_tmp.extend(num*[laber])
    
  labers = list(map(int,laber_tmp))
  
  feats_dat = np.array(feats)
  labers_dat = np.array(labers)
  
  feats_train, feats_test, labels_train, labels_test = train_test_split(feats_dat, labers_dat, test_size=0.1, random_state=0)
         
  test = [feats_test, labels_test]

  # parameters
  ed = 100
  ps = 5
  fd = 128
  dp = 0.5
  lr = 0.001
  para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
          'drop_out': dp, 'learning_rate': lr}
  
  dir = '/home/user/Desktop/yyf_data/MLBP/MLBP/'
  test_my(test, para, dir)