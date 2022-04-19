import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

directory = os.path.normpath(r'F:/FlowHet_RxnDist')
sample = "ket0.1ph3.1"
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
imagesize = (df.nz[sample],df.ny[sample],df.nx[sample])

subdir = "/Menke2017/ket0.1ph3.1/baseline/Batch100"
structure_file = directory + subdir + "/final_structure_rxnsim.raw"
structure_sim = np.fromfile(structure_file,dtype=np.dtype('uint8'))
structure_sim = structure_sim.reshape((imagesize[0]+20,imagesize[1],imagesize[2]))
structure_sim = structure_sim[9:imagesize[0]+9,:,:]


subdir = "/Menke2017/ket0.1ph3.1/structures"
structure_file = directory + subdir + "/ket0.1ph3.1_modelbinary_duringreaction_3.raw"
structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
structure = structure.reshape(imagesize)

print('hi')