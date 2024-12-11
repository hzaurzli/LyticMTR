import os
import time
import argparse
import numpy as np
from pathlib import Path
dir = 'CNN_base'
Path(dir).mkdir(exist_ok=True)
from sklearn.model_selection import train_test_split


def Training(tr_data, tr_label):

    from train import train_main # load my training function

    train = [tr_data, tr_label]

    threshold = 0.5
    
    train_main(train, dir)
    
    print(train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("-iX", "--input_X", required=True, type=str, help="feature table (txt,'\t')")
    parser.add_argument("-iy", "--input_y", required=True, type=str, help="laber file (txt, '\t'")
    Args = parser.parse_args()
    
    f = open(Args.input_X)
    g = open(Args.input_y)
    
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
    
    Training(feats_dat, labers_dat)
  
    