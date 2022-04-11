#The GLA is applied in the ROM_LSTM surrogate model
#Two choices are provided in this script (either with POD or POD AE) 
import sys
import math
import random
import copy
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl
#import imageio
import pickle
import tensorflow as tf
from tensorflow import keras
import time
import scipy
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from smt.sampling_methods import LHS
from adao import adaoBuilder


fields_ens = np.load('data_post/fields_oil_1traj.npy')
################################################################
#velocity data already compressed
vx_ens = np.load('data_POD/vx_oil_POD30_1traj_30_ens.npy')
vy_ens = np.load('data_POD/vy_oil_POD30_1traj_30_ens.npy')
vz_ens = np.load('data_POD/vz_oil_POD30_1traj_30_ens.npy')

################################################################
#define input and output sequence size
n_memory = 30
n_prediction = 30
latent_dim = 120

input_data_alpha = np.load('LSTM_data/alphaoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_input.npy')
output_data_alpha = np.load('LSTM_data/alphaoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_output.npy')

input_data_vx = np.load('LSTM_data/vxoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_input.npy')
output_data_vx = np.load('LSTM_data/vxoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_output.npy')

input_data_vy = np.load('LSTM_data/vyoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_input.npy')
output_data_vy = np.load('LSTM_data/vyoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_output.npy')

input_data_vz = np.load('LSTM_data/vzoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_input.npy')
output_data_vz = np.load('LSTM_data/vzoil_52fi30dense_POD30_'+str(n_memory)+'to'+str(n_prediction)+'_output.npy')



u_pod_vx = np.load("data_post/u_pod_vx_full_30.npy")[:,:30]
u_pod_vy = np.load("data_post/u_pod_vy_full_30.npy")[:,:30]
u_pod_vz = np.load("data_post/u_pod_vz_full_30.npy")[:,:30]

############################################################################
#u_pod_sample = np.load("data_post/u_pod_random_H_10000.npy")
u_pod_sample = np.load("data_post/u_pod_choice_H_60000.npy")


input_data = np.concatenate((input_data_alpha,input_data_vx),axis = 2)
input_data = np.concatenate((input_data,input_data_vy),axis = 2)
input_data = np.concatenate((input_data,input_data_vz),axis = 2)

output_data = np.concatenate((output_data_alpha,output_data_vx),axis = 2)
output_data = np.concatenate((output_data,output_data_vy),axis = 2)
output_data = np.concatenate((output_data,output_data_vz),axis = 2)



####################################################################
method = 'POD'
#method = 'POD_AE'
#method = 'CDAE'
increment = True
scaled = False
run_DA = False

DA_index = []

for i in range(570,999,150):
     DA_index += list(range(i,i+30))
#DA_index = list(range(540,570)) + list(range(840,870)) 

################################################################
generation = 999
 #number of simulation time steps in total
prediction_start = 300
look_back = 30
pred_length = 30
latent_dim = 120
###########################################################
fields_ens = np.load('data_post/fields_oil_1traj.npy')
fields_compress = np.load('data_post/field_POD_reconstructed .npy' )

if method == 'POD':
    u_pod = np.load("data_post/u_pod_dense_1traj_full.npy")[:,:30]
    filename = 'ML_model/LSTM_POD_velo30_1traj_diff_true.h5'
    #filename = 'ML_model/LSTM_POD_velo30_1traj_diff_10to10.h5'
    #filename = 'ML_model/LSTM_POD_velo30_1traj_diff_true.h5'

if method == 'POD_AE':
    u_pod = np.load("data_post/u_pod_dense_1traj_full.npy")[:,:799]
    filename = 'ML_model/LSTM_POD_AE_30_1traj_diff30to30.h5'
    encoder = keras.models.load_model('ML_model/encoder_POD_AE_'+str(latent_dim)+'.h5')
    decoder = keras.models.load_model('ML_model/decoder_POD_AE_'+str(latent_dim)+'.h5')
    
    encoder_vx = keras.models.load_model('ML_model/encoder_POD_AE_30_vx.h5')
    decoder_vx = keras.models.load_model('ML_model/decoder_POD_AE_30_vx.h5')
    encoder_vy = keras.models.load_model('ML_model/encoder_POD_AE_30_vy.h5')
    decoder_vy = keras.models.load_model('ML_model/decoder_POD_AE_30_vy.h5')   
    encoder_vz = keras.models.load_model('ML_model/encoder_POD_AE_30_vz.h5')
    decoder_vz = keras.models.load_model('ML_model/decoder_POD_AE_30_vz.h5')
    
LSTM_POD_model = keras.models.load_model(filename)




###############################################################
# constructing obsrevation sample
#L = list(np.load('data_post/obs_sample_90000.npy'))
n_LHS = 1000
poly_degree = 4 
ratio = 0.5
#encoder_sample = keras.models.load_model('ML_model/encoder_POD_AE_30_random_H_10000.h5')
#decoder_sample = keras.models.load_model('ML_model/decoder_POD_AE_30_random_H_10000.h5')
encoder_sample = keras.models.load_model('ML_model/encoder_POD_AE_30_choice_H_60000.h5')
decoder_sample = keras.models.load_model('ML_model/decoder_POD_AE_30_choice_H_60000.h5')
#H = np.load('data_post/random_H.npy')
H = scipy.sparse.load_npz('data_post/H_rdn_choice_60000.npz').todense()
fields_ens_sample = np.load('data_post/choiceH_POD_reconstructed_60000.npy')
sample_ens_encoded = encoder_sample.predict(fields_ens_sample)
polynomial_features= PolynomialFeatures(degree=poly_degree)


#############################################################
#initializing variables

POD_series = np.zeros((look_back,latent_dim)) # sequence of input

POD_predict = np.zeros((look_back,latent_dim))# sequence of output prediction

DL_error = []

pod_error_MC = []

time_LSTM = []

norm_field = []

############################################################

LSTM_error = []

prediction_error = []

obspod_error = []

DA_error = []

###############################################################
#debug
POD_initial_vect = []

POD_initial_predict_vect = []

true_traj = []
pred_traj = []
coord = 700
###############################################################

for i in range(0,generation):
    
    print(i)
    
    #input_vect_all = input_data[i,:]
    field = fields_ens[i,:]
    vx_initial = vx_ens[i,:]
    vy_initial = vy_ens[i,:]
    vz_initial = vz_ens[i,:]
    
    # alpha_initial = input_data_alpha[i,0,:]
    # vx_initial = input_data_vx[i,0,:]
    # vy_initial = input_data_vy[i,0,:]
    # vz_initial = input_data_vz[i,0,:]
    
    field_initial = np.copy(field)
    
    norm_field.append(np.linalg.norm(field))  
    
##################################################################################

    
    if i<prediction_start:
        
        #########################################################################
        #LSTM
        #########################################################################
        POD_series.shape = (look_back,latent_dim)
        POD_series = np.roll(POD_series, -1, axis=0)
        field_initial.shape = (field_initial.size,1)
        #########################################################################
        #POD
        if method == 'POD':
            POD_initial = np.dot (u_pod.T,field_initial)
            
            POD_initial.shape = (POD_initial.size,1)
            
            field_reconstruction_predict = np.dot(u_pod, POD_initial)
            
            vect_all = list(POD_initial)+list(vx_initial)+list(vy_initial)+list(vz_initial)
            # vect_all = list(alpha_initial)+list(vx_initial)+list(vy_initial)+list(vz_initial)
            # vect_all = input_data[i,0,:]
            POD_series[look_back-1,:] = np.array(vect_all).ravel() 
        #####################################################################    
                #POD
        if method == 'POD_AE':
            
            encoded = encoder.predict(fields_compress[i,:].reshape(1,-1,1))
            POD_initial = encoded.reshape(encoded.size,1)
            POD_initial.shape = (POD_initial.size,1)

            field_reconstruction_predict = np.dot(u_pod, POD_initial)
            
            vect_all = list(POD_initial)+list(vx_initial)+list(vy_initial)+list(vz_initial)
            # vect_all = list(alpha_initial)+list(vx_initial)+list(vy_initial)+list(vz_initial)
            # vect_all = input_data[i,0,:]
            POD_series[look_back-1,:] = np.array(vect_all).ravel() 
       #####################################################################   
       
    else:
        if (i-prediction_start)%look_back == 0:

            POD_series.shape = (1,look_back,latent_dim)
            
            t = time.time()
            POD_p = LSTM_POD_model.predict(POD_series)
            POD_p.shape = (look_back,latent_dim)
            
            if increment == True:
                
                POD_p[0,:] = POD_series[0,look_back-1,:]+POD_p[0,:]
                
                POD_p = np.cumsum(POD_p,axis=0)

            time_LSTM.append(time.time()-t)
            POD_p.shape = (look_back,latent_dim)
            
        #####################################################################
        index_in_p = ((i-prediction_start)%look_back)
        
        POD_initial_predict = POD_p[index_in_p ,:]
        
        ######################################################################                
        #########################################################################
        
        #########################################################################
        
        POD_initial_predict.shape = (POD_initial_predict.size,1)
        
        #########################################################################
        #POD

        field_reconstruction_predict = np.dot(u_pod, POD_initial_predict[:30,:])
        
        POD_initial = np.dot (u_pod.T,field_initial)
        vect_all = list(POD_initial)+list(vx_initial)+list(vy_initial)+list(vz_initial)
        # vect_all = input_data[i,0,:]
        # vect_all = output_data[i,0,:]
        
        vect_all = np.array(vect_all).ravel() 
        POD_initial_vect.append(vect_all[0:latent_dim])
        POD_initial_predict_vect.append(POD_initial_predict[0:latent_dim])

        #########################################################################
        #DA
        ########################################################################
        if method == 'POD':
        #DA
            if run_DA == True and i in DA_index:
                x_b = np.copy(POD_initial_predict[:30].ravel()).reshape(1,-1)
                x_t = POD_initial.reshape(1,-1)
                y = sample_ens_encoded[i,:].reshape(1,-1)
                xlimits = np.array([[x_b[0,i]-abs(x_b[0,i])*ratio ,x_b[0,i]+abs(x_b[0,i])*ratio ] for i in range(x_b.size)])
                sampling = LHS(xlimits=xlimits,random_state=10)
    
                LHS_ens = sampling(n_LHS).reshape(n_LHS,-1,1)
                                
                field_background_ens = np.dot(u_pod, LHS_ens.reshape(-1,n_LHS)).reshape(field.size,-1)
                
                obs_full_ens = np.dot(H, field_background_ens)
                sample_compressed = np.dot(u_pod_sample.T, obs_full_ens)
                
                y_train = encoder_sample.predict(sample_compressed.T)
                polynomial_features= PolynomialFeatures(degree=poly_degree)
                LHS_ens.shape = (n_LHS,30)
                x_poly = polynomial_features.fit_transform(LHS_ens)
                
                model = LinearRegression()
                model.fit(x_poly, y_train)
                y_poly_pred = model.predict(x_poly)
    
                
                def Poly_operator(x):
                    x_poly_test = polynomial_features.fit_transform(x.reshape(1,-1))
                    y = model.predict(x_poly_test)
                    return y.ravel()
                
                case = adaoBuilder.New()
                case.set( 'AlgorithmParameters', Algorithm='3DVAR',
                         Parameters = {"Minimizer" : "LBFGSB","MaximumNumberOfSteps":25,
                                       "CostDecrementTolerance":1.e-2,
                                       "StoreSupplementaryCalculations":["CostFunctionJ","CurrentState",
                                        "SimulatedObservationAtOptimum",
                                        "SimulatedObservationAtBackground",
                                        "JacobianMatrixAtBackground",
                                        "JacobianMatrixAtOptimum",
                                        "KalmanGainAtOptimum",
                                        "APosterioriCovariance"]
                                       } )
                case.set( 'Background',          Vector=x_b)
                case.set( 'BackgroundError',     ScalarSparseMatrix=10.0 )
                case.set( 'Observation',         Vector=y )
                case.set( 'ObservationError',    ScalarSparseMatrix=1.0 )
                case.set( 'ObservationOperator', OneFunction = Poly_operator)
                case.set( 'Observer',            Variable="Analysis", Template="ValuePrinter" )
                
                #case.setObserver(Variable="CostFunctionJ",Template="ValuePrinter")
                #case.setObserver(Variable="CostFunctionJo",Template="ValuePrinter")
                #case.setObserver(Variable="CostFunctionJb",Template="ValuePrinter")
                case.execute()
                
                x_a = case.get("Analysis")[-1]
                
                POD_initial_predict[:30] = x_a.reshape(-1,1)
                
                print('(Poly_operator(x_b)-y)',np.linalg.norm(Poly_operator(x_b)-y))
                print('(Poly_operator(x_a)-y)',np.linalg.norm(Poly_operator(x_a)-y))
                print('(Poly_operator(x_t)-y)',np.linalg.norm(Poly_operator(x_t)-y))
                print('x_b-x_t',np.linalg.norm(x_b-x_t))
                print('x_a-x_t',np.linalg.norm(x_a-x_t))        
        #########################################################################
        ########################################################################
        if method == 'POD AE':
        #DA
            if run_DA == True and i in DA_index:
                x_b = np.copy(POD_initial_predict[:30].ravel()).reshape(1,-1)
                x_t = POD_initial.reshape(1,-1)
                y = sample_ens_encoded[i,:].reshape(1,-1)
                xlimits = np.array([[x_b[0,i]-abs(x_b[0,i])*ratio ,x_b[0,i]+abs(x_b[0,i])*ratio ] for i in range(x_b.size)])
                sampling = LHS(xlimits=xlimits,random_state=10)
    
                LHS_ens = sampling(n_LHS).reshape(n_LHS,-1,1)
                decoded_LHS = decoder.predict(LHS_ens)
                
                field_background_ens = np.dot(u_pod, decoded_LHS.reshape(-1,n_LHS)).reshape(field.size,-1)
                
                obs_full_ens = np.dot(H, field_background_ens)
                sample_compressed = np.dot(u_pod_sample.T, obs_full_ens)
                
                y_train = encoder_sample.predict(sample_compressed.T)
                polynomial_features= PolynomialFeatures(degree=poly_degree)
                LHS_ens.shape = (n_LHS,30)
                x_poly = polynomial_features.fit_transform(LHS_ens)
                
                model = LinearRegression()
                model.fit(x_poly, y_train)
                y_poly_pred = model.predict(x_poly)
    
                
                def Poly_operator(x):
                    x_poly_test = polynomial_features.fit_transform(x.reshape(1,-1))
                    y = model.predict(x_poly_test)
                    return y.ravel()
                
                case = adaoBuilder.New()
                case.set( 'AlgorithmParameters', Algorithm='3DVAR',
                         Parameters = {"Minimizer" : "LBFGSB","MaximumNumberOfSteps":25,
                                       "CostDecrementTolerance":1.e-2,
                                       "StoreSupplementaryCalculations":["CostFunctionJ","CurrentState",
                                        "SimulatedObservationAtOptimum",
                                        "SimulatedObservationAtBackground",
                                        "JacobianMatrixAtBackground",
                                        "JacobianMatrixAtOptimum",
                                        "KalmanGainAtOptimum",
                                        "APosterioriCovariance"]
                                       } )
                case.set( 'Background',          Vector=x_b)
                case.set( 'BackgroundError',     ScalarSparseMatrix=10.0 )
                case.set( 'Observation',         Vector=y )
                case.set( 'ObservationError',    ScalarSparseMatrix=1.0 )
                case.set( 'ObservationOperator', OneFunction = Poly_operator)
                case.set( 'Observer',            Variable="Analysis", Template="ValuePrinter" )
                
                #case.setObserver(Variable="CostFunctionJ",Template="ValuePrinter")
                #case.setObserver(Variable="CostFunctionJo",Template="ValuePrinter")
                #case.setObserver(Variable="CostFunctionJb",Template="ValuePrinter")
                case.execute()
                
                x_a = case.get("Analysis")[-1]
                
                POD_initial_predict[:30] = x_a.reshape(-1,1)
                
                print('(Poly_operator(x_b)-y)',np.linalg.norm(Poly_operator(x_b)-y))
                print('(Poly_operator(x_a)-y)',np.linalg.norm(Poly_operator(x_a)-y))
                print('(Poly_operator(x_t)-y)',np.linalg.norm(Poly_operator(x_t)-y))
                print('x_b-x_t',np.linalg.norm(x_b-x_t))
                print('x_a-x_t',np.linalg.norm(x_a-x_t))        
        #########################################################################
        #POD AE
        #########################################################################             
        POD_series.shape = (1,look_back,latent_dim)
                    
        POD_series = np.roll(POD_series, -1, axis=0)
        
        POD_series[0,look_back-1,:] = POD_initial_predict.ravel()
        
        POD_series.shape = (look_back,latent_dim)
        
    ################################################################################
    LSTM_error.append(np.linalg.norm(field_reconstruction_predict.ravel() -field.ravel() ))
    