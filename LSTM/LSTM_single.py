import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Activation,Dropout,RepeatVector
from tensorflow.python.keras.layers import BatchNormalization, TimeDistributed, LayerNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras import backend as K

import keras
from keras.layers import LeakyReLU

import tensorflow as tf
#tf.enable_eager_execution()

from tensorflow.keras import optimizers
# check scikit-learn version
# check scikit-learn version
import pandas as pd
import os
import json
import pickle
###############################################################

def scale_data(train_data):
    input_data = np.load('LSTM_data/alphaoil_52fi30dense_POD_AE30_input_pred1.npy')
    max_speed = input_data.max()
    min_speed = input_data.min()
    train_scaled = (train_data - min_speed) / (max_speed - min_speed)
    return train_scaled


def rescale_data(train_data):
    
    input_data = np.load('LSTM_data/alphaoil_52fi30dense_POD_AE30_input_pred1.npy')
    max_speed = input_data.max()
    min_speed = input_data.min()
    
    train_rescaled = train_data*(max_speed - min_speed) + min_speed
    
    return train_rescaled
    
################################################################
#define input and output sequence size
n_memory = 10
n_prediction = 10
latent_dim = 30


input_data = np.load('LSTM_data/alphaoil_52fi30dense_POD30_input.npy')
output_data = np.load('LSTM_data/alphaoil_52fi30dense_POD30_output.npy')
# ################################################################
#diff
output_data[:,1:,:] = np.diff(output_data, axis=1)

output_data[:,0,:] = output_data[:,0,:] - input_data[:,n_prediction-1,:]

###############################################################################
#preprocessing 
#########################################################################

test_index = list(set(range(1,939,5)))
train_index = list(set(range(1,939)) - set(range(1,939,5)))


##########################################################################
train_input = input_data[train_index,:,:]

train_output = output_data[train_index,:]

test_input = input_data[test_index,:,:]

true_test_output = output_data[test_index,:]

X1 = train_input

Y1 = train_output

X2 = test_input

Y2 = true_test_output
#######################################################################
hidden_size=50

input_sample = input_data.shape[0]  #for one sample

output_sample = output_data.shape[0]

use_dropout=True

model = Sequential()

model.add(LSTM(hidden_size,input_shape=(n_memory,latent_dim)))

model.add(RepeatVector(n_prediction))

model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(LayerNormalization(axis=1 , center=True , scale=True))

model.add(Dense(100,activation='relu'))

model.add(Dense(100,activation='relu'))


model.add(TimeDistributed(Dense(latent_dim)))

model.add(LeakyReLU(alpha=0.3))

##################################################################
#training
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
history = model.fit(input_data , output_data, validation_split=0.01, epochs=2000,batch_size=32,verbose=2)

# ####################################################################
# # evalutation in the latent space

PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)

plt.plot(PredTestSet[:,0,0],train_output[:,0,0],'o', color='blue',markersize=5)
plt.plot(train_output[:,0,0],train_output[:,0,0], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 1, 1st eig')
plt.show()

plt.plot(PredValSet[:,0,0],true_test_output[:,0,0],'o', color='blue',markersize=5)
plt.plot(true_test_output[:,0,0],true_test_output[:,0,0], color='r',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 1, 1st eig')
plt.show()


plt.plot(PredValSet[:,0,1],true_test_output[:,0,1],'o', color='blue',markersize=5)
plt.plot(true_test_output[:,0,1],true_test_output[:,0,1], color='r',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 1, 2nd eig')
plt.show()

plt.plot(PredValSet[:,0,4],true_test_output[:,0,4],'o', color='blue',markersize=5)
plt.plot(true_test_output[:,0,4],true_test_output[:,0,4], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 1, 5th eig')
plt.show()

# ####################################################################""

plt.plot(PredValSet[:,9,0],true_test_output[:,9,0],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,9,0],PredValSet[:,9,0], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 10, 1st eig')
plt.show()

plt.plot(PredValSet[:,9,1],true_test_output[:,9,1],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,9,1],PredValSet[:,9,1], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 10, 2nd eig')
plt.show()


plt.plot(PredValSet[:,9,4],true_test_output[:,9,4],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,9,4],PredValSet[:,9,4], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)
plt.title('dt = 10, 5th eig')

plt.show()

############################################################################

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Fire_loss.pdf', format='pdf')
plt.show()
#
model.save('ML_model/LSTM_POD_30_1traj.h5')

