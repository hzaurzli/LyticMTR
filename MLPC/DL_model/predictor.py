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

 
def predict(X_test, h5_model,res):
 # 1.loading weight and structure (model)
    h5_model_path = h5_model
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

    output_file = res
    count = 0
    with open(output_file, 'w') as f:
        f.write('ID' + '\t' + 'G-P' + '\t' + 'MES' + '\t' + 'G-G' + '\t' + 'Other' + '\t' + 'P-P' + '\n')
        for i in functions:
            f.write(feat_name[count] + '\t' + '\t'.join(i) + '\n')
            count += 1
    f.close()
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-iX", "--input_X", required=True, type=str, help="feature table (txt,'\t')")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="model path (.h5)")
    parser.add_argument("-r", "--res", required=True, type=str, help="result file")
    Args = parser.parse_args()  
    
    f = open(Args.input_X)
    h5_model = Args.model_path
    res = Args.res      
  
    max_seq_len = 500
    feats = []
    feat_name = []
    for i in f:
      feat_name.append(i.strip().split('\t')[0])
      feat_property = i.strip().split('\t')[1:6]
      feat_other = i.strip().split('\t')[6:]
      feat_property_padding = [0] * int(max_seq_len - len(feat_property))
      feat_property.extend(feat_property_padding)
      feat_property.extend(feat_other)
      feat_tmp = list(map(float,feat_property))
      feats.append(feat_tmp)
    
    feats_dat = np.array(feats)
    predict(feats_dat,h5_model,res)
