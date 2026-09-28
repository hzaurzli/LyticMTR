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
import tensorflow as tf


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
    load_my_model = load_model(h5_model_path, custom_objects={'keras.backend': keras.backend, 'expand_dims': expand_dims})
    
    heatmaps = np.zeros([1, int(500)], dtype=float)

    y_pred = list(load_my_model.predict(X_test)[0])
    
    #y_pred = load_my_model.predict(X_test)  # 需要先进行预测
    
    # 对模型中的变量进行初始化,否则会报错
    init = tf.global_variables_initializer()
    sess = K.get_session()
    sess.run(init)

    ret = load_my_model.output[0, np.argmax(y_pred)]  # 更简洁的写法
    
    # 获取双向循环层
    last_conv_layer = load_my_model.get_layer("bidirectional_1")
    fm = last_conv_layer.output
    
    # 计算梯度
    grads = K.gradients(ret, last_conv_layer.output)[0]
    
    # 对于循环神经网络，通常在时间维度上求平均
    pooled_grads = K.mean(grads, axis=(0, 1))  # 假设输出形状为 (batch, timesteps, features)
    
    # 创建计算函数
    iterate = K.function([load_my_model.input], [pooled_grads, last_conv_layer.output[0]])
    
    # 初始化heatmaps（需要在循环外初始化）
    heatmaps = np.array([]).reshape(0, X_test.shape[1])  # 假设时间步长为 X_test.shape[1]
    
    # 对每个测试样本计算heatmap
    for sample_idx in range(len(X_test)):
        # 获取当前样本的预测
        sample_pred = load_my_model.predict(X_test[sample_idx:sample_idx+1])
        ret = load_my_model.output[0, np.argmax(sample_pred)]
        
        # 重新定义计算图（因为ret依赖于具体样本）
        grads = K.gradients(ret, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1))
        iterate = K.function([load_my_model.input], [pooled_grads, last_conv_layer.output[0]])
        
        # 计算梯度和特征图
        pooled_grads_value, conv_layer_output_value = iterate([X_test[sample_idx:sample_idx+1]])
        
        # 应用梯度权重
        for i in range(pooled_grads_value.shape[0]):  # 特征维度
            conv_layer_output_value[:, i] *= pooled_grads_value[i]
        
        # 创建heatmap
        heatmap = np.mean(conv_layer_output_value, axis=-1)  # 沿特征维度平均
        heatmap = np.squeeze(heatmap)  # 移除不必要的维度
        
        # 存储heatmap
        if len(heatmaps) == 0:
            heatmaps = heatmap[np.newaxis, :]
        else:
            heatmaps = np.vstack([heatmaps, heatmap[np.newaxis, :]])
    
    # 计算平均heatmap
    mean_heatmap = np.mean(heatmaps, axis=0)
    mean_heatmap = np.maximum(mean_heatmap, 0)
    mean_heatmap /= np.max(mean_heatmap) + 1e-10  # 避免除零错误
    
    print("Final heatmap shape:", mean_heatmap.shape)

    print(mean_heatmap)
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
          if i < int(mean_heatmap.shape[0]):
              f.write(str(i+1)+","+seq[i]+","+str(mean_heatmap[i])+"\n")
          else:
              f.write(str(i+1)+","+seq[i]+",0\n")
      f.close()
    
            
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
    
    
    
    
