import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
import argparse,os



def get_tail_label(df):
    """
    Give tail label colums of the given target dataframe

    args
    df: pandas.DataFrame, target label df whose tail label has to identified

    return
    tail_label: list, a list containing column name of all the tail label
    """
    columns = df.columns
    n = len(columns)
    irpl = np.zeros(n)
    for column in range(n):
        irpl[column] = df[columns[column]].value_counts()[1]
    irpl = max(irpl) / irpl
    mir = np.average(irpl)
    tail_label = []
    for i in range(n):
        if irpl[i] > mir:
            tail_label.append(columns[i])
    return tail_label


def get_index(df):
    """
    give the index of all tail_label rows
    args
    df: pandas.DataFrame, target label df from which index for tail label has to identified

    return
    index: list, a list containing index number of all the tail label
    """
    tail_labels = get_tail_label(df)
    index = set()
    for tail_label in tail_labels:
        sub_index = set(df[df[tail_label] == 1].index)
        index = index.union(sub_index)
    return list(index)


def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels

    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe

    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    X_sub = X[X.index.isin(index)].reset_index(drop=True)
    y_sub = y[y.index.isin(index)].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n - 1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


if __name__ == '__main__':
    """
    main function to use the MLSMOTE
    """
    parser = argparse.ArgumentParser(description="MLSMOTE")
    parser.add_argument("-iX", "--input_X", required=True, type=str, help="feature table (txt,'\t')")
    parser.add_argument("-iy", "--input_y", required=True, type=str, help="laber file (txt, '\t')")
    parser.add_argument("-o", "--output", required=True, type=str, help="output path")
    parser.add_argument("-n", "--n_sample", required=True, type=int, help="number of newly generated sample")
    Args = parser.parse_args()
    
    path = os.path.abspath(Args.output)
    print(path)
    
    X_file = Args.input_X
    f = open(X_file)
    with open(path + '/X_tmp.txt','w') as w:
      for i in f:
        line = '\t'.join(i.strip().split('\t')[1:]) + '\n'
        w.write(line)
    w.close()
    
    X_data = pd.read_table(path + '/X_tmp.txt', low_memory=False, sep = '\t',header = None)
    X_df = pd.DataFrame(X_data)
    X = X_df
    
    y_file = Args.input_y
    y_data = pd.read_table(y_file, low_memory=False, sep = '\t',header = None)
    y_df = pd.DataFrame(y_data)
    y = y_df

    X_sub, y_sub = get_minority_instace(X, y)  # Getting minority instance of that dataframe
    X_sub.to_csv(path + "/X_sub.txt", index=False, sep='\t')
    y_sub.to_csv(path + "/y_sub.txt", index=False, sep='\t')
    
    print('finish minority instace')
    
    X_res, y_res = MLSMOTE(X_sub, y_sub, Args.n_sample)  # Applying MLSMOTE to augment the dataframe
    re_index = ['sample_' + str(re) for re in range(1, len(X_res) + 1)]
    X_res.index = re_index
    y_res.index = re_index
  
    X_res.to_csv(path + "/X_res.txt", header=0, index=True, sep='\t')
    y_res.to_csv(path + "/y_res.txt", header=0, index=True, sep='\t')
    os.remove(path + '/X_tmp.txt')
