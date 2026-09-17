import os
import time
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model


def build_linear_proj(in_dim, name="proj"):
    inp = Input(shape=(in_dim,), name=f"{name}_in")
    out = Dense(1, use_bias=False, name=name)(inp)
    model = Model(inp, out, name=f"{name}_model")
    return model


def seq_linear(data, encode_seq, max_len, proj_model=None):
    if proj_model is None:
        proj_model = build_linear_proj(in_dim=21, name="seq_proj")

    seq_linear_feature = []
    for i in data:
        soh = [encode_seq[j] for j in i]
        arr = np.asarray(soh, dtype="float32")

        reduced = proj_model.predict(arr, verbose=0).flatten()

        if len(reduced) < max_len:
            pad = np.zeros(max_len - len(reduced), dtype=reduced.dtype)
            reduced = np.concatenate([reduced, pad])

        seq_linear_feature.append(reduced.tolist())

    return seq_linear_feature, proj_model


def struct_linear(struct_a, encode, max_len, proj_model=None):
    if proj_model is None:
        proj_model = build_linear_proj(in_dim=8, name="struct_proj")

    struct_linear_feature = []
    for i in struct_a:
        soh = [encode[j] for j in i]
        arr = np.asarray(soh, dtype="float32")

        reduced = proj_model.predict(arr, verbose=0).flatten()
        if len(reduced) < max_len:
            pad = np.zeros(max_len - len(reduced), dtype=reduced.dtype)
            reduced = np.concatenate([reduced, pad])

        struct_linear_feature.append(reduced.tolist())

    return struct_linear_feature, proj_model


def fasta2dict(fasta_name):
    with open(fasta_name) as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n', '')
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n', '')
    return fa_dict


if __name__ == '__main__':
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Feature engine")
    parser.add_argument("-f",  "--fasta",     required=True, type=str, help="protein sequence")
    parser.add_argument("-s",  "--ss",        required=True, type=str, help="ss8 format, secondary structure")
    parser.add_argument("-p",  "--property",  required=True, type=str, help="property table ('\\t')")
    parser.add_argument("-rf", "--res_feat",  required=True, type=str, help="feature matrix file (output)")
    Args = parser.parse_args()

    input_path_1 = Args.fasta
    input_path_2 = Args.ss
    input_path_3 = Args.property
    res_feat     = Args.res_feat

    fa_seq    = fasta2dict(input_path_1)
    fa_struct = fasta2dict(input_path_2)

    data     = []
    struct_a = []
    seq_a    = []

    # for sturcture
    encode = {
        'H': [0,0,0,0,0,0,0,1], 'G': [0,0,0,0,0,0,1,0],
        'I': [0,0,0,0,0,1,0,0], 'E': [0,0,0,0,1,0,0,0],
        'B': [0,0,0,1,0,0,0,0], 'T': [0,0,1,0,0,0,0,0],
        'S': [0,1,0,0,0,0,0,0], 'C': [1,0,0,0,0,0,0,0]
    }
    
    # for sequence
    encode_seq = {
        'X': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        'A': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
        'C': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        'D': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        'E': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        'F': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        'G': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        'H': [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        'I': [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        'K': [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        'L': [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
        'M': [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        'N': [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        'P': [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'Q': [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'R': [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'S': [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'T': [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'V': [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'W': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        'Y': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    }

    for key in fa_seq:
        data.append(fa_seq[key])
    for key in fa_seq:
        struct_a.append(fa_struct[key])

    seq_lenmax = 500

    seq_linear_feature, seq_proj_model = seq_linear(data, encode_seq, seq_lenmax)
    struct_linear_feature, struct_proj_model = struct_linear(struct_a, encode, seq_lenmax)

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
    property_feat.close()

    property_seq_pca = []
    for i in range(len(seq_linear_feature)):
        property_seq_pca.append(
            property_all_lis[i] + seq_linear_feature[i] + struct_linear_feature[i]
        )

    with open(res_feat, 'w', newline='') as f:
        for i in property_seq_pca:
            new_i = list(map(str, i))
            line = '\t'.join(new_i) + '\n'
            f.write(line)
    f.close()