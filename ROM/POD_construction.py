# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt 

from scipy.sparse import diags

from scipy.sparse.csgraph import reverse_cuthill_mckee

import scipy.sparse as sp

import Ofpp

#optional: loading data from openfoam dataset
mesh = Ofpp.FoamMesh('.')
alpha_oil = Ofpp.parse_internal_field('data-um52fi30-dense-vel/'+"{:.2f}".format(1*0.01)+'/alpha.oil')
    
fields_ens = np.zeros((3,alpha_oil.size))

for i in range(1,1000):
    #loading your data
    alpha_oil = Ofpp.parse_internal_field('data-um52fi30-dense-vel/'+"{:.2f}".format(i*0.01)+'/alpha.oil')
    fields_ens = np.concatenate((fields_ens, alpha_oil.reshape(1,alpha_oil.size)),axis = 0)    
fields_ens = fields_ens[1:,]


#########################################################################
u_pod, s_pod, v_pod = np.linalg.svd(fields_ens.T, full_matrices=False)
# u_pod contains all POD basis and s_pod contains the eigenvalues
#



