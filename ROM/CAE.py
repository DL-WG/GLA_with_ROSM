import pandas as pd
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt 

from scipy.sparse import diags
from scipy.sparse.csgraph import reverse_cuthill_mckee

import scipy.sparse as sp

from tensorflow.python.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D,Cropping1D, AveragePooling1D,Dense,Flatten,Reshape,Dropout
from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf
from tensorflow import keras
import pickle
from tensorflow.python.keras.layers import LeakyReLU

fields_ens = np.load('data/fields_oil_1traj.npy')
test_index = list(set(range(1,999,5)))
train_index = list(set(range(1,fields_ens.shape[0])) - set(range(1,999,5)))

# reorder the meshes with graph-based algorithms
cm_order = list(np.load('data_post/cm_order.npy'))
fields_ens = fields_ens[:,cm_order]

#train_index = range(0,threshold)
#test_index = range(threshold,fields_ens.shape[0])

x_train = fields_ens[train_index,:]
x_train = x_train.reshape(x_train.shape[0], 180000,1)
x_test = fields_ens[test_index,:]
x_test = x_test.reshape(x_test.shape[0], 180000,1)


# define the encoder
input_sig = Input(batch_shape=(None,180000,1))

x2 = Conv1D(4,8, activation='relu',padding='same',dilation_rate=2)(input_sig)
#x2 = LeakyReLU(alpha=0.3)(x2)
x2 = Dropout(0.1)(x2)
x3 = MaxPooling1D(5)(x2)

x4 = Conv1D(4,8,activation='relu',padding='same',dilation_rate=2)(x3)
#x4 = LeakyReLU(alpha=0.3)(x4)
x4 = Dropout(0.1)(x4)
x5 = MaxPooling1D(5)(x4)

x6 = Conv1D(1,8, activation='relu',padding='same',dilation_rate=2)(x5)
x6 = LeakyReLU(alpha=0.3)(x6)
x6 = Dropout(0.1)(x6)
x7 = MaxPooling1D(5)(x6)

x8 = AveragePooling1D()(x7)

flat = Flatten()(x8)

encoded = Dense(30)(flat)

encoder = Model(input_sig, encoded)

# define the decoder
decoder_input= Input(shape=(30,))

d1 = Dense(720)(decoder_input)
d2 = Reshape((720,1))(d1)
#d3 = Conv1D(1,8,strides=1, activation='relu', padding='same')(d2)
#d4 = UpSampling1D(2)(d3)

d5 = Conv1D(1,8,strides=1, activation='relu', padding='same')(d2)
d6 = UpSampling1D(10)(d5)

d7 = Conv1D(1,8,strides=1, activation='relu', padding='same')(d6)
d8 = UpSampling1D(5)(d7)

d9 = Conv1D(1,8,strides=1, padding='same')(d8)
d10 = UpSampling1D(5)(d9)

decoded = Conv1D(1,1,strides=1, padding='same')(d10)
decoded = LeakyReLU(alpha=0.3)(decoded)
decoder = Model(decoder_input, decoded)

#combine the encoder and the decoder
decoder_input= Input(shape=(30,))

decoded = Dense(30)(decoder_input)
decoded = LeakyReLU(alpha=0.3)(decoded)

decoded = Dense(100)(decoded)
decoded = LeakyReLU(alpha=0.3)(decoded)

decoded = Dense(180000)(decoded)
decoded = LeakyReLU(alpha=0.3)(decoded)
decoded = Reshape((180000,1))(decoded)
decoder = Model(decoder_input, decoded)


auto_input = Input(batch_shape=(None,180000,1))
encoded = encoder(auto_input)
decoded = decoder(encoded)

autoencoder = Model(auto_input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(x_train, x_train, epochs=1000, batch_size=16,shuffle=True, validation_data=(x_test, x_test))