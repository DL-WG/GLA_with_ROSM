# -*- coding: utf-8 -*-
# preprocessing for forming the training/testing dataset(already compressed) for LSTM
# any compressed data (POD, CAE or POD AE) can be used for the LSTM surrogate mdoel

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

# total time steps of fire simulations
generation = 999

latent_dim = 30

n_memory = 10 #number of look back steps

n_prediction = 10 #number of prediction steps

fields_1_sim = np.zeros((1,latent_dim))

for i in range(generation):
        
    field_pod = np.load('data_POD/encoded_POD_AE'+str(latent_dim)+'_1traj_vz_'+str(i)+'.npy') 
    fields_1_sim = np.concatenate((fields_1_sim,field_pod.reshape(1,field_pod.size)),axis = 0)
      
fields_1_sim = fields_1_sim[1:,:]
    
input_data = np.zeros((1,n_memory,latent_dim))
output_data = np.zeros((1,n_prediction,latent_dim))

current_simulation = np.copy(fields_1_sim)
    #    

for j in range(generation-n_memory-n_prediction):

    input_lstm = current_simulation[j:(j+n_memory),:]
    
    input_data = np.concatenate((input_data,input_lstm.reshape(1,n_memory,latent_dim)),axis = 0)

    output_lstm = current_simulation[(j+n_memory):(j+n_memory+n_prediction),:]

    output_data = np.concatenate((output_data,output_lstm.reshape(1,n_prediction,latent_dim)),axis = 0)
    
print('input_data.shape',input_data.shape)
print('output_data.shape',output_data.shape)
    
input_data = input_data[1:,:]
output_data = output_data[1:,:]


np.save('LSTM_data/vx_52fi30dense_POD_AE10to10_input.npy',input_data)
np.save('LSTM_data/vx_52fi30dense_POD_AE10to10_output.npy',output_data)