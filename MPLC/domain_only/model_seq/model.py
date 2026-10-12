from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, merge, Lambda
from keras.layers import Flatten, Dense, Activation, BatchNormalization, GRU, LSTM
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import Bidirectional
from keras.backend import expand_dims
import numpy as np


def slice(x, w1, w2):
    """ Define a tensor slice function
    """
    return x[:,w1:w2]


def expand_dim(x):
    xa = expand_dims(x, axis=-1)
    return xa


def BiGRU_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    # 张量的切片也要写为层的形式,函数式会报错
    input_seq = Lambda(slice, arguments={'w1': 0, 'w2': length})(main_input)
    length_seq = int(input_seq.shape[1]) # 这里注意转换一下数据类型,否则会报"TypeError: unhashable type: 'Dimension'"
   
    ####### sequence
    #x = Embedding(output_dim=ed, input_dim=21, input_length=length_seq)(input_seq)
    x = Lambda(expand_dim)(input_seq)

    a_seq = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_seq)

    b_seq = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_seq)

    c_seq = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_seq)
    
    
    # apool_property,bpool_property, cpool_property,apool_seq, bpool_seq, cpool_seq, apool_struct,bpool_struct, cpool_struct
    merge = Concatenate(axis=-1)([apool_seq, bpool_seq, cpool_seq])
    merge = Dropout(dp)(merge)


    x = Bidirectional(GRU(50, return_sequences=True))(merge) # cpu对应函数为GRU, gpu对应函数为CuDNNGRU
    x = Flatten()(x)
    x = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x)

    # output = Dense(out_length, activation='sigmoid', name='output')(x)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model