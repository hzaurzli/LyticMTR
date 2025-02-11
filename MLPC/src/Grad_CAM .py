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
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import argparse


def fasta2dict(fasta_name):    
  with open(fasta_name) as fa:
      fa_dict = {}
      for line in fa:
          line = line.replace('\n','')
          if line.startswith('>'):
              seq_name = line[1:]
              fa_dict[seq_name] = ''
          else:
              fa_dict[seq_name] += line.replace('\n','')
  return fa_dict
  
 
def predict(X_test, h5_model, ss8_file, res_path):

    h5_model_path = h5_model
    load_my_model = load_model(h5_model_path, custom_objects={'keras.backend': keras.backend, 'expand_dims': expand_dims}) # 需要传一个custom_objects参数,将自定义的层添加进去
    
    heatmaps = np.zeros([1, int(500)], dtype=float)

    y_pred = list(load_my_model.predict(X_test)[0])
    ret = load_my_model.output[0, y_pred.index(max(y_pred))]
    last_conv_layer = load_my_model.get_layer("bidirectional_1")
    fm = last_conv_layer.output
    grads = K.gradients(ret, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1))
    iterate = K.function([load_my_model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([X_test])
    for i in range(pooled_grads.shape[0]):  # 梯度和特征图加权
      conv_layer_output_value[:, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = heatmap[np.newaxis, :]
    
    heatmaps = np.append(heatmaps, heatmap, axis=0)
    
    heatmap = heatmaps.mean(axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap[np.newaxis, :]
    
    print(heatmap)
    
    fa = fasta2dict(ss8_file)

    for key in fa:
      seq_dir = os.path.abspath(res_path)
      if key.strip().split(' ')[0].startswith('s'):
        name = key.strip().split(' ')[0].split('|')[1]
        seq_dir = seq_dir + '/' + name + ".csv"
      else:
        name = key.strip().split(' ')[0]
        seq_dir = seq_dir + '/' + name + ".csv"
      
      f = open(seq_dir, "w")
      seq = fa[key]
      f.write("No.,residue,weight\n")
      for i in range(len(seq)):
          if i < len(heatmap[0]):
              f.write(str(i+1)+","+seq[i]+","+str(heatmap[0][i])+"\n")
          else:
              f.write(str(i+1)+","+seq[i]+",0\n")
      f.close()
      
      print(heatmap)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grad CAM")
    parser.add_argument("-iX", "--input_X", required=True, type=str, help="feature table (txt,'\t')")
    parser.add_argument("-m", "--model_path", required=True, type=str, help="model path (.h5)")
    parser.add_argument("-s", "--ss8_file", required=True, type=str, help="ss8 file")
    parser.add_argument("-r", "--res_path", required=True, type=str, help="result path")
    Args = parser.parse_args()  
    
    f = open(Args.input_X)
    h5_model = Args.model_path
    ss8_file = Args.ss8_file
    res_path = Args.res_path
  
    max_seq_len = 500
    
    for i in f:
      feats = []
      feat_other = i.strip().split('\t')[506:]
      feat_tmp = list(map(float,feat_other)) 
      #feat_other = i.strip().split('\t')[506:]
      #feat_tmp = list(map(float,feat_other))
      feats.append(feat_tmp)
      feats_dat = np.array(feats)
      predict(feats_dat,h5_model,ss8_file,res_path)
    
    
    
    
