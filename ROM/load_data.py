#transfer openfoam data to numpy array
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from scipy.sparse import diags
from scipy.sparse.csgraph import reverse_cuthill_mckee
import scipy.sparse as sp
import Ofpp


U_ens = np.zeros((1,3,alpha_oil.size))
for i in range(1,800):
    print(i)    
    file_name = 'newdata/data-dense-dw-2e-4/data-um52fi45-dense/'+"{:.2f}".format(i*0.01)+'/U.oil'    
    a_file = open(file_name, "r")
    list_of_lines = a_file.readlines()
    
    U_current = np.zeros((1,3,alpha_oil.size))
    
    for j in range(22,180022):
        line_vect = np.array(str.split(list_of_lines[j][1:-2]))
        U_current[0,:,j-22] = line_vect.copy()
    U_ens = np.concatenate((U_ens, U_current.reshape(1,3,alpha_oil.size)),axis = 0)
U_ens = U_ens[1:,:,:]
print(U_ens.shape)
np.save("data_post/Uoil_ens_um52fi45_dense.npy",U_ens)