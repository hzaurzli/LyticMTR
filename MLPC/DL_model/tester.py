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
import argparse


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


def predict(X_test, y_test, y_laber, para, h5_model):
    
    adam = Adam(lr=para['learning_rate']) # adam optimizer
    # 1.loading weight and structure (model)
    h5_model_path = h5_model
    load_my_model = load_model(h5_model_path, custom_objects={'keras.backend': keras.backend, 'expand_dims': expand_dims}) # 需要传一个custom_objects参数,将自定义的层添加进去
    print("Prediction is in progress")

    # 2.predict
    score = load_my_model.predict(X_test)

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
    
    pred = []
    for i in functions:
      laber_tmp = ''.join(i)
      if laber_tmp == '10000':
          pred.append(0)
      elif laber_tmp == '01000':
          pred.append(1)
      elif laber_tmp == '00100':
          pred.append(2)
      elif laber_tmp == '00010':
          pred.append(3)
      elif laber_tmp == '00001':
          pred.append(4)

    "========================================"
    macro_precision,macro_recall,macro_f1,matthews_corr,accuracy,absolute_true,absolute_false = evaluate(score_laber, y_test, pred, y_laber)
    print("Prediction is done")
    print('macro precision:', macro_precision)
    print('macro recall:', macro_recall)
    print('macro f1:', macro_f1)
    print('matthews corr:', matthews_corr)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')
    
    
    output_file = './result_perf.txt'
    with open(output_file, 'w') as f:
      f.write('macro precision: '+ str(macro_precision) + '\n')
      f.write('macro recall: ' + str(macro_recall) + '\n')
      f.write('macro f1: ' + str(macro_f1) + '\n')
      f.write('matthews corr: ' + str(matthews_corr) + '\n')
      f.write('accuracy: ' + str(accuracy) + '\n')
      f.write('absolute_true: ' + str(absolute_true) + '\n')
      f.write('absolute_false: ' + str(absolute_false) + '\n')
    f.close()
      


def run_tester(test, y_laber, para):
    # step1: preprocessing
    test[1] = keras.utils.to_categorical(test[1])
    test[0], temp = catch(test[0], test[1])
    temp[temp > 1] = 1
    test[1] = temp
        
    # step2:predict
    predict(test[0], test[1], y_laber, para, h5_model)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-iX", "--input_X", required=True, type=str, help="feature table (txt,'\t')")
    parser.add_argument("-iy", "--input_y", required=True, type=str, help="laber file (txt, '\t')")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="model path (.h5)")
    Args = parser.parse_args()  
      
    f = open(Args.input_X)
    g = open(Args.input_y)
    h5_model = Args.model_path
    
    # f = open('/home/user/Desktop/yyf_data/laber/random_sample/final_input.txt')
    # g = open('/home/user/Desktop/yyf_data/laber/random_sample/laber.txt')
    
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
    
    test = [feats_dat, labers_dat]
  
    # parameters
    ed = 100
    ps = 5
    fd = 128
    dp = 0.5
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}
    
    run_tester(test, labers, para)