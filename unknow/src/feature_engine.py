import os
import numpy as np
from pathlib import Path
from keras.models import load_model
import argparse
import os
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import os
import tensorflow as tf
import keras
import csv
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import Flatten, Dense, Activation, BatchNormalization, CuDNNGRU, CuDNNLSTM
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import Bidirectional
from sklearn.decomposition import PCA 



data = ['LLENGAIFVTSG', 'VTTNGPPNGKHNDKHTYVEC', 'SAGMIPAEPGEA']
struct_a = ['HHBHBGTSICEEBH','GTGTSIIICEBHHB','ECISTTGBBH']
encode = {'H': [0,0,0,0,0,0,0,1] , 'G': [0,0,0,0,0,0,1,0], 'I': [0,0,0,0,0,1,0,0], 'E': [0,0,0,0,1,0,0,0], 
          'B': [0,0,0,1,0,0,0,0], 'T': [0,0,1,0,0,0,0,0], 'S': [0,1,0,0,0,0,0,0], 'C': [1,0,0,0,0,0,0,0]}


def PadEncode(data, max_len):

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e = []
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0]*(max_len-length)
        data_e.append(elemt)

    return data_e
    
 
def struct_pca(struct_a,encode,max_len):
    pca_feature = []
    for i in struct_a:
      soh = []
      for j in i:
        soh.append(encode[j])
        
      pca = PCA(n_components=1)      
      array = np.asarray(soh)
      reduced_x = pca.fit_transform(array)
      reduced_y = reduced_x.tolist()
      reduced_z = []
      for n in reduced_y:
        reduced_z.append(n[0])
        
      elemt = []
      if len(reduced_z) < max_len:
        elemt += [0]*(max_len-len(reduced_z))
        reduced_z.extend(elemt)
          
      pca_feature.append(reduced_z)
      
    return pca_feature
    

def fasta2dict(fasta_name):    
    with open(fasta_name) as fa:
        fa_dict = {}
        for line in fa:
            # 去除末尾换行符
            line = line.replace('\n','')
            if line.startswith('>'):
                # 去除 > 号
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace('\n','')
    return fa_dict

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Lysin finder")
    parser.add_argument("-f", "--fasta", required=True, type=str, help="protein sequence")
    parser.add_argument("-s", "--ss", required=True, type=str, help="ss8 format,secondary structure")
    parser.add_argument("-fo", "--out_fasta", required=True, type=str, help="protein sequence output by length filtering")
    parser.add_argument("-so", "--out_ss", required=True, type=str, help="ss8 format,secondary structure output by length filtering")
    parser.add_argument("-li", "--len_min", required=False, default=100, type=float, help="upper proteins length")
    parser.add_argument("-lq", "--len_max", required=False, default=500, type=float, help="lower proteins length")
    parser.add_argument("-rf", "--res_feat", required=True, type=str, help="feature matrix file")
    Args = parser.parse_args()
    
    # executing the main function
    #input_path_1 = '/home/user/Desktop/laber/amidase_struct_90/amidase.fasta'
    #input_path_2 = '/home/user/Desktop/laber/amidase_struct_90/amidase.ss8'
    #output_path_1 = '/home/user/Desktop/laber/amidase_struct_90/amidase_100_500.fasta'
    #output_path_2 = '/home/user/Desktop/laber/amidase_struct_90/amidase_100_500.ss8'
    #res_feat = '/home/user/Desktop/laber/feat_amidase.csv'
    
    input_path_1 = Args.fasta
    input_path_2 = Args.ss
    output_path_1 = Args.out_fasta
    output_path_2 = Args.out_ss
    res_feat = Args.res_feat
    len_min = Args.len_min
    len_max = Args.len_max
    
    
    fa_seq = fasta2dict(input_path_1)
    fa_struct = fasta2dict(input_path_2)
    
    data = []
    struct_a = []
    encode = {'H': [0,0,0,0,0,0,0,1] , 'G': [0,0,0,0,0,0,1,0], 'I': [0,0,0,0,0,1,0,0], 'E': [0,0,0,0,1,0,0,0], 
              'B': [0,0,0,1,0,0,0,0], 'T': [0,0,1,0,0,0,0,0], 'S': [0,1,0,0,0,0,0,0], 'C': [1,0,0,0,0,0,0,0]}
    
    with open(output_path_1,'w') as w:  
      for key in fa_seq:
        if len(fa_seq[key]) < len_max and len(fa_seq[key]) > len_min:
          line = '>' + fa_seq[key] + '\n' + fa_seq[key] + '\n'
          w.write(line)
          data.append(fa_seq[key])
    w.close()
    
    with open(output_path_2,'w') as w:
      for key in fa_struct:
        if len(fa_struct[key]) < len_max and len(fa_struct[key]) > len_min:
          line = '>' + fa_struct[key] + '\n' + fa_struct[key] + '\n'
          w.write(line)
          struct_a.append(fa_struct[key])
    w.close()
    
    
    seq_lenmax = len(max(data, key=len, default=''))
    seq_feature = PadEncode(data,seq_lenmax)
    pca_feature = struct_pca(struct_a,encode,seq_lenmax)
    
    seq_pca = []
    for i in range(0,len(seq_feature)):
      seq_pca.append(seq_feature[i] + pca_feature[i])
      
    with open(res_feat,'w',newline='') as f:
      for i in seq_pca:
        new_i = list(map(str,i))
        line = ','.join(new_i)
        f.write(line)
    f.close()
    
   
        
