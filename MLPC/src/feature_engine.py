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
    parser.add_argument("-p", "--property", required=True, type=str, help="property")
    parser.add_argument("-rf", "--res_feat", required=True, type=str, help="feature matrix file")
    parser.add_argument("-rfs", "--res_feat_seq", required=True, type=str, help="sequence feature matrix file")
    parser.add_argument("-rfp", "--res_feat_property", required=True, type=str, help="property feature matrix file")
    parser.add_argument("-rft", "--res_feat_struct", required=True, type=str, help="secondary structurefeature matrix file")
    Args = parser.parse_args()
    
    # executing the main function
    #input_path_1 = '/home/user/Desktop/laber/glycosidase_struct_90/glycosidase_100_500.fasta'
    #input_path_2 = '/home/user/Desktop/laber/glycosidase_struct_90/glycosidase_100_500.ss8'
    #input_path_3 = '/home/user/Desktop/laber/property/glycosidase_100_500.txt'
    #res_feat = '/home/user/Desktop/laber/feat_glycosidase.csv'
    #res_feat_seq = '/home/user/Desktop/laber/feat_seq_glycosidase.csv'
    #res_feat_property = '/home/user/Desktop/laber/feat_property_glycosidase.csv'
    #res_feat_struct = '/home/user/Desktop/laber/feat_struct_glycosidase.csv'
    
    input_path_1 = Args.fasta
    input_path_2 = Args.ss
    input_path_3 = Args.property  
    res_feat = Args.res_feat
    res_feat_seq = Args.res_feat_seq
    res_feat_property = Args.res_feat_property
    res_feat_struct = Args.res_feat_struct

    fa_seq = fasta2dict(input_path_1)
    fa_struct = fasta2dict(input_path_2)
    
    data = []
    struct_a = []
    encode = {'H': [0,0,0,0,0,0,0,1] , 'G': [0,0,0,0,0,0,1,0], 'I': [0,0,0,0,0,1,0,0], 'E': [0,0,0,0,1,0,0,0], 
              'B': [0,0,0,1,0,0,0,0], 'T': [0,0,1,0,0,0,0,0], 'S': [0,1,0,0,0,0,0,0], 'C': [1,0,0,0,0,0,0,0]}
 
    for key in fa_seq:
      data.append(fa_seq[key])
      
    for key in fa_seq:
      struct_a.append(fa_struct[key])
          
    seq_lenmax = len(max(data, key=len, default=''))
    seq_feature = PadEncode(data,seq_lenmax)
    pca_feature = struct_pca(struct_a,encode,seq_lenmax)
    
    
    property_feat = open(input_path_3)
    property_all_lis = []
    next(property_feat)
    for i in property_feat:
      item = i.strip().split('\t')
      property_lis = []
      property_lis.append(item[0])
      property_lis.append(item[3])
      property_lis.append(item[4])
      property_lis.append(item[5])
      property_lis.append(item[6])
      property_lis.append(item[7])
      property_all_lis.append(property_lis) 
      
    property_seq_pca = []
    for i in range(0,len(seq_feature)):
      property_seq_pca.append(property_all_lis[i] + seq_feature[i] + pca_feature[i])
      
    with open(res_feat,'w',newline='') as f:
      for i in property_seq_pca:
        new_i = list(map(str,i))
        line = '\t'.join(new_i) + '\n'
        f.write(line)
    f.close()
    
    
    with open(res_feat_seq,'w',newline='') as f:
      for i in seq_feature:
        new_i = list(map(str,i))
        line = '\t'.join(new_i) + '\n'
        f.write(line)
    f.close()
    
    
    with open(res_feat_property,'w',newline='') as f:
      for i in property_all_lis:
        new_i = list(map(str,i))
        line = '\t'.join(new_i) + '\n'
        f.write(line)
    f.close()
    
    
    with open(res_feat_struct,'w',newline='') as f:
      for i in pca_feature:
        new_i = list(map(str,i))
        line = '\t'.join(new_i) + '\n'
        f.write(line)
    f.close()
    
   
   
        
