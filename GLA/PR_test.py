# in this script we test the local poly nomial regression for the GLA algorithm
import operator
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from tensorflow.python.keras.models import Model,Sequential
import tensorflow as tf
import keras
from smt.sampling_methods import LHS
from adao import adaoBuilder


def invreg(x):
    return 1/(x+0.5)


#set initial parameters
n_LHS = 1000 #the number of samplings in PR
poly_degree = 4 
ratio = 0.3 # ratio of the LHS samplings

fields_ens = np.load('data/fields_oil_1traj.npy')
u_pod = np.load("data_post/u_pod_dense_1traj_full.npy")
#u_pod_sample = np.load("data_post/u_pod_choice_H_30000_invreg.npy")
u_pod_sample = np.load("data_post/u_pod_choice_H_30000_power2.npy")

#encoder_sample = keras.models.load_model('ML_model/encoder_POD_AE_30_choice_H_30000_invreg.h5')
#decoder_sample = keras.models.load_model('ML_model/decoder_POD_AE_30_choice_H_30000_invreg.h5')

encoder_sample = keras.models.load_model('ML_model/encoder_POD_AE_30_choice_H_30000_2.h5')
decoder_sample = keras.models.load_model('ML_model/decoder_POD_AE_30_choice_H_30000_2.h5')
#H = np.load('data_post/random_H.npy')
H = scipy.sparse.load_npz('data_post/H_rdn_choice_30000.npz').todense()
#fields_ens_sample = np.load('data_post/choiceH_POD_reconstructed_30000_invreg.npy')

fields_ens_sample = np.load('data_post/choiceH_POD_reconstructed_30000_2.npy')
sample_ens_encoded = encoder_sample.predict(fields_ens_sample)
polynomial_features= PolynomialFeatures(degree=poly_degree)

encoder = keras.models.load_model('ML_model/encoder_POD_AE_'+str(30)+'.h5')
decoder = keras.models.load_model('ML_model/decoder_POD_AE_'+str(30)+'.h5')
#############################################
#x_true = encoder.predict(field.reshape(1,-1,1))
################################################
field_compress = np.dot(u_pod.T,fields_ens[300,:].reshape(-1,1)).reshape(1,-1)

x_true = encoder.predict(field_compress.reshape(1,-1,1))

correct_compress = np.dot(u_pod.T,fields_ens[300,:].reshape(-1,1)).reshape(1,-1)

x_t = encoder.predict(correct_compress.reshape(1,-1,1))

y = sample_ens_encoded[300,:]

field_corret = fields_ens[500,:]


###################################################################################
# LHS sampling x_true
xlimits = np.array([[x_true[0,i]-abs(x_true[0,i])*ratio ,x_true[0,i]+abs(x_true[0,i])*ratio ] for i in range(x_true.size)])
sampling = LHS(xlimits=xlimits,random_state=10)
LHS_ens = sampling(n_LHS).reshape(n_LHS,-1,1)

###########################################################

full_PC = decoder.predict(LHS_ens).T
field_background_ens = np.dot(u_pod, full_PC)

#field_background_ens = invreg(field_background_ens)

field_background_ens = np.power(field_background_ens,2)

obs_full_ens = np.dot(H, field_background_ens)
sample_compressed = np.dot(u_pod_sample.T, obs_full_ens)
               
y_train = encoder_sample.predict(sample_compressed.T)
###########################################################
##########################################

LHS_ens.shape = (n_LHS,30)

polynomial_features= PolynomialFeatures(degree=poly_degree)

x_poly = polynomial_features.fit_transform(LHS_ens)

model = LinearRegression()
model.fit(x_poly, y_train)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y_train,y_poly_pred))
#r2 = r2_score(y,y_poly_pred)
print(rmse)
#print(r2)
plt.plot(y_poly_pred[:,0],y_train[:,0],'go')
plt.plot(y_train[:,0],y_train[:,0],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_1.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred[:,1],y_train[:,1],'go')
plt.plot(y_train[:,1],y_train[:,1],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_2.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred[:,2],y_train[:,2],'go')
plt.plot(y_train[:,2],y_train[:,2],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_3.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred[:,3],y_train[:,3],'go')
plt.plot(y_train[:,3],y_train[:,3],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_4.eps', format='eps')
plt.show()
plt.close()


ratio = 0.6
###################################################################################
# LHS sampling x_true
xlimits_test = np.array([[x_true[0,i]-abs(x_true[0,i])*ratio ,x_true[0,i]+abs(x_true[0,i])*ratio ] for i in range(x_true.size)])
sampling_test = LHS(xlimits=xlimits_test,random_state=12)
LHS_ens_test = sampling_test(n_LHS).reshape(n_LHS,-1,1)

full_PC_test = decoder.predict(LHS_ens_test).T
field_background_ens_test = np.dot(u_pod, full_PC_test)

###########################################################
#field_background_ens_test = np.dot(u_pod, LHS_ens_test.reshape(-1,n_LHS)).reshape(180000,-1)

#field_background_ens_test = invreg(field_background_ens_test)

field_background_ens_test = np.power(field_background_ens_test,2)

obs_full_ens_test = np.dot(H, field_background_ens_test)
sample_compressed_test = np.dot(u_pod_sample.T, obs_full_ens_test)
               
y_train_test = encoder_sample.predict(sample_compressed_test.T)
###########################################################
##########################################

LHS_ens_test.shape = (n_LHS,30)

polynomial_features= PolynomialFeatures(degree=poly_degree)

x_poly_test = polynomial_features.fit_transform(LHS_ens_test)
y_poly_pred_test = model.predict(x_poly_test)

rmse = np.sqrt(mean_squared_error(y_train_test,y_poly_pred_test))
#r2 = r2_score(y,y_poly_pred)
print(rmse)
#print(r2)
plt.plot(y_poly_pred_test[:,0],y_train_test[:,0],'ro')
plt.plot(y_train_test[:,0],y_train_test[:,0],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
plt.savefig('figure/NN_poly_large_300_LV_1.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred_test[:,1],y_train_test[:,1],'ro')
plt.plot(y_train_test[:,1],y_train_test[:,1],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
plt.savefig('figure/NN_poly_large_300_LV_2.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred_test[:,2],y_train_test[:,2],'ro')
plt.plot(y_train_test[:,2],y_train_test[:,2],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
plt.savefig('figure/NN_poly_large_300_LV_3.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred_test[:,3],y_train_test[:,3],'ro')
plt.plot(y_train_test[:,3],y_train_test[:,3],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_test_300_LV_4.eps', format='eps')
plt.savefig('figure/NN_poly_large_300_LV_4.eps', format='eps')
plt.show()
plt.close()
