from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout, merge, Lambda
from keras.layers import Flatten, Dense, Activation, BatchNormalization, GRU
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

# 定义为层,而非函数
def base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001
     
    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    # 张量的切片也要写为层的形式,函数式会报错
    input_seq = Lambda(slice, arguments={'w1': 500, 'w2': 1000})(main_input)
    length_seq = int(input_seq.shape[1]) # 这里注意转换一下数据类型,否则会报"TypeError: unhashable type: 'Dimension'"
    # 张量的切片也要写为层的形式,函数式会报错
    input_property = Lambda(slice, arguments={'w1': 0, 'w2': 500})(main_input)
    # 张量的切片也要写为层的形式,函数式会报错
    input_struct = Lambda(slice, arguments={'w1': 1000, 'w2': length})(main_input)
 
    ####### sequence
    #x = Embedding(output_dim=ed, input_dim=21, input_length=length_seq)(input_seq)
    x = Lambda(expand_dim)(input_seq)

    a_seq = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_seq)

    b_seq = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_seq)

    c_seq = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_seq)
    
    ####### property
    #main_input_property_tmp = keras.backend.expand_dims(main_input_property, axis=-1)
    y = Lambda(expand_dim)(input_property) # 要先定义扩展维度的函数expand_dim,然后用Lambda定义层

    a_property = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    apool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_property)

    b_property = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    bpool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_property)

    c_property = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    cpool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_property)
    
    ####### struct
    #main_input_struct_tmp = keras.backend.expand_dims(main_input_struct, axis=-1)
    z = Lambda(expand_dim)(input_struct) # 要先定义扩展维度的函数expand_dim,然后用Lambda定义层
    
    a_struct = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    apool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_struct)

    b_struct = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    bpool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_struct)

    c_struct = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    cpool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_struct)
    
    # apool_property,bpool_property, cpool_property,apool_seq, bpool_seq, cpool_seq, apool_struct,bpool_struct, cpool_struct
    merge = Concatenate(axis=-1)([apool_property,bpool_property, cpool_property,apool_seq, bpool_seq, cpool_seq, apool_struct,bpool_struct, cpool_struct])
    merge = Dropout(dp)(merge)

    x_merge = Flatten()(merge)
    x_merge = Dense(fd, activation='relu', name='FC1', W_regularizer=l2(l2value))(x_merge)
    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x_merge)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model


def BiGRU_base(length, out_length, para):

    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='float32', name='main_input')
    # 张量的切片也要写为层的形式,函数式会报错
    input_seq = keras.layers.Lambda(slice, arguments={'w1': 500, 'w2': 1000})(main_input)
    length_seq = int(input_seq.shape[1]) # 这里注意转换一下数据类型,否则会报"TypeError: unhashable type: 'Dimension'"
    # 张量的切片也要写为层的形式,函数式会报错
    input_property = keras.layers.Lambda(slice, arguments={'w1': 0, 'w2': 500})(main_input)
    # 张量的切片也要写为层的形式,函数式会报错
    input_struct = keras.layers.Lambda(slice, arguments={'w1': 1000, 'w2': length})(main_input)
 
    ####### sequence
    #x = Embedding(output_dim=ed, input_dim=21, input_length=length_seq)(input_seq)
    x = keras.layers.Lambda(expand_dim)(input_seq)

    a_seq = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_seq)

    b_seq = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_seq)

    c_seq = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool_seq = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_seq)
    
    ####### property
    #main_input_property_tmp = keras.backend.expand_dims(main_input_property, axis=-1)
    y = keras.layers.Lambda(expand_dim)(input_property) # 要先定义扩展维度的函数expand_dim,然后用Lambda定义层

    a_property = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    apool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_property)

    b_property = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    bpool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_property)

    c_property = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(y)
    cpool_property = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_property)
    
    ####### struct
    #main_input_struct_tmp = keras.backend.expand_dims(main_input_struct, axis=-1)
    z = keras.layers.Lambda(expand_dim)(input_struct) # 要先定义扩展维度的函数expand_dim,然后用Lambda定义层
    
    a_struct = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    apool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a_struct)

    b_struct = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    bpool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b_struct)

    c_struct = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(z)
    cpool_struct = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c_struct)
    
    # apool_property,bpool_property, cpool_property,apool_seq, bpool_seq, cpool_seq, apool_struct,bpool_struct, cpool_struct
    merge = keras.layers.Concatenate(axis=-1)([apool_property,bpool_property, cpool_property,apool_seq, bpool_seq, cpool_seq, apool_struct,bpool_struct, cpool_struct])
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
