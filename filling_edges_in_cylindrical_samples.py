import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = os.getcwd()+'/data'

df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)


samples = ['Sil_HetA_High_Scan1','Sil_HetA_Low_Scan1','Sil_HetB_High_Scan1','Sil_HetB_Low_Scan1']

for sample in samples:
    structure_file = directory + "/" + sample + ".raw"
    structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
    structure = structure.reshape((df.loc[sample]['nz'],df.loc[sample]['ny'],df.loc[sample]['nx']))
    structure[structure == 0] = 2 #changing corners to solid ID
    structure[structure == 1] = 0 #changing pore ID to 0
    structure[structure == 2] = 1 #changing solid ID to 1
    structure.astype('uint8').tofile(directory +"/" + sample + '_structure.raw')
