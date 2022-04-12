# -*- coding: utf-8 -*-
#apply GLA for an inverse problem with an initial guess
#a forward model is required

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

import time
import pickle
import keras# check scikit-learn version

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize
from sklearn.preprocessing import PolynomialFeatures

from smt.sampling_methods import LHS
from adao import adaoBuilder


###########################################################
#parameter of GLA
n_LHS = 1000 # number of samplings
poly_degree = 4
ratio = 0.5 #prior ratio of GLA sampling
polynomial_features= PolynomialFeatures(degree=poly_degree)

# define the backgrond state, here we give a simple example of a vector of dimension 4
x_b = np.array([[250.    , 175.    ,  63.2959, 299.111 ]])
y = .... #your observations 

xlimits = np.array([[x_b[0,i]-abs(x_b[0,i])*ratio ,x_b[0,i]+abs(x_b[0,i])*ratio ] for i in range(x_true.size)])
sampling = LHS(xlimits=xlimits,random_state=10)
LHS_ens = sampling(n_LHS).reshape(n_LHS,-1,1)


LHS_ens.shape = (n_LHS,4)

polynomial_features= PolynomialFeatures(degree=poly_degree)
x_poly = polynomial_features.fit_transform(LHS_ens)
y_train = KN_model.predict(LHS_ens)

model = LinearRegression()
model.fit(x_poly, y_train)
y_poly_pred = model.predict(x_poly)


####################################################
# test the polynomial regression accuracy

plt.plot(y_poly_pred[:,0],y_train[:,0],'go')
plt.plot(y_train[:,0],y_train[:,0],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_1.eps', format='eps')
plt.show()
plt.close()

plt.plot(y_poly_pred[:,15],y_train[:,15],'go')
plt.plot(y_train[:,15],y_train[:,15],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_2.eps', format='eps')
plt.show()
plt.close()


plt.plot(y_poly_pred[:,25],y_train[:,25],'go')
plt.plot(y_train[:,25],y_train[:,25],'b')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('truth',fontsize = 16)
#plt.savefig('figure/NN_poly_train_300_LV_2.eps', format='eps')
plt.show()
plt.close()

###############################################
#define the local surrogate function
def f(x):
    x_poly_test = polynomial_features.fit_transform(x.reshape(1,-1))
    y = model.predict(x_poly_test)
    return y.ravel()

# define the background error covariance matrix
Back_matrix = np.diag([100000,10,100000,10])

case = adaoBuilder.New()
case.set( 'AlgorithmParameters', Algorithm='3DVAR',
         Parameters = {"Minimizer" : "LBFGSB","MaximumNumberOfSteps":100,
                       "CostDecrementTolerance":1.e-5,
                       "StoreSupplementaryCalculations":["CostFunctionJ","CurrentState",
                        "SimulatedObservationAtOptimum",
                        "SimulatedObservationAtBackground",
                        "JacobianMatrixAtBackground",
                        "JacobianMatrixAtOptimum",
                        "KalmanGainAtOptimum",
                        "APosterioriCovariance"]
                       } )
case.set( 'Background',          Vector=x_b)
case.set( 'BackgroundError',     Matrix=Back_matrix )
case.set( 'Observation',         Vector=y )
case.set( 'ObservationError',    ScalarSparseMatrix=1.0 )
case.set( 'ObservationOperator', OneFunction = f)
case.set( 'Observer',            Variable="Analysis", Template="ValuePrinter" )

#case.setObserver(Variable="CostFunctionJ",Template="ValuePrinter")
#case.setObserver(Variable="CostFunctionJo",Template="ValuePrinter")
#case.setObserver(Variable="CostFunctionJb",Template="ValuePrinter")
case.execute()

x_a = case.get("Analysis")[-1]

#x_a_reconstucted = decoder.predict(x_a.reshape(1,30))

#x_a_reconstucted = np.dot(u_pod,x_a_reconstucted.reshape(-1,1))

print('xb-xt', np.linalg.norm(x_b.ravel()-x_t.ravel()))
print('xa-xt', np.linalg.norm(x_a.ravel()-x_t.ravel()))

print('Hxb-y', np.linalg.norm(f(x_b).ravel()-y.ravel()))
print('Hxa-y', np.linalg.norm(f(x_a).ravel()-y.ravel()))