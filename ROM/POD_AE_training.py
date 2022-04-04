# -*- coding: utf-8 -*-
#these script perform a dense autoencoder with already preprcossed data (including POD projection)
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
import keras
from keras.layers import LeakyReLU
import pickle


fields_ens = np.load('data_post/vx_POD_reconstructed.npy')

#seperating training and testing dataset
test_index = list(set(range(1,999,5)))
train_index = list(set(range(1,fields_ens.shape[0])) - set(range(1,999,5)))

###########################################################

x_train = fields_ens[train_index,:]

x_train = x_train.reshape(x_train.shape[0], 999,1)

x_test = fields_ens[test_index,:]

x_test = x_test.reshape(x_test.shape[0], 999,1)


############################################################################
#############################################################################
# latent with velocity
input_img = keras.Input(shape=(999,))

encoded = Dense(128)(input_img)

encoded  = LeakyReLU(alpha=0.3)(encoded)

encoded = Dense(30)(encoded)

encoded  = LeakyReLU(alpha=0.3)(encoded)
encoder = keras.Model(input_img, encoded)
decoder_input= Input(shape=(30,))


decoded = Dense(128)(decoder_input)
decoded  = LeakyReLU(alpha=0.3)(decoded )

decoded  = Dense(999)(decoded )
decoded  = LeakyReLU(alpha=0.3)(decoded )


decoder = keras.Model(decoder_input, decoded)

auto_input = keras.Input(shape=(999,))
encoded = encoder(auto_input)
decoded = decoder(encoded)

model = keras.Model(auto_input, decoded)

#############################################################################

model.compile(optimizer='adam', loss='mse')

history = model.fit(x_train, x_train, epochs=3000, batch_size=16,shuffle=True, validation_data=(x_test, x_test))

zz = model.predict(x_test[10,:,:].reshape(1,-1,1))
plt.plot(zz.ravel()[1:],'r',label = 'prediction')
plt.plot(x_test[10,:],label = 'truth')
plt.xlabel('eigenvalues')
plt.legend()

#save the trained model
encoder.save('ML_model/encoder_POD_AE_100.h5')
decoder.save('ML_model/decoder_POD_AE_100.h5')
model.save('ML_model/POD_AE_100_CM_nolist_model_1trajcorrect.h5')

encoder.save('ML_model/encoder_POD_AE_30_vz.h5')
decoder.save('ML_model/decoder_POD_AE_30_vz.h5')
model.save('ML_model/POD_AE_100_CM_nolist_model_1trajcorrect.h5')

encoder.save('ML_model/encoder_POD_AE_30_sample_90000.h5')
decoder.save('ML_model/decoder_POD_AE_30_sample_90000.h5')


encoder.save('ML_model/encoder_POD_AE_30_random_H.h5')
decoder.save('ML_model/decoder_POD_AE_30_random_H.h5')