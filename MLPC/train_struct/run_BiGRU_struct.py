import os
import time
import argparse
import numpy as np
from pathlib import Path
dir = 'BiGRU_base'
Path(dir).mkdir(exist_ok=True)


def Training(tr_data, tr_label):

    from trainer import trainer_main # load my training function

    train = [tr_data, tr_label]

    threshold = 0.5
    
    trainer_main(train, dir)
    
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
      feat_other = i.strip().split('\t')[506:]
      feats.append(feat_other)
      
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
