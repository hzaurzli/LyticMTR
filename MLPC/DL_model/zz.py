import os
import time
import numpy as np
from pathlib import Path
dir = 'BiGRU_base'
Path(dir).mkdir(exist_ok=True)
from sklearn.model_selection import train_test_split


def TrainAndTest(tr_data, tr_label, te_data, te_label):

    from train import train_main # load my training function

    train = [tr_data, tr_label]
    test = [te_data, te_label]

    threshold = 0.5
    model_num = 9  # model number
    test.append(threshold)
    
    train_main(train, test, model_num, dir)
    
    print(train)


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
  print(feats_dat)
  print(labers_dat)
  
  feats_train, feats_test, labels_train, labels_test = train_test_split(feats_dat, labers_dat, test_size=0.3, random_state=0)
  
  TrainAndTest(feats_train, labels_train, feats_test, labels_test)


  