#velocity distribution
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

#customize these for each sample
# sample_descriptor = "menke_ketton"
# imagesize =(922,902,911)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
# sample_descriptor = "beadpack"
# imagesize =(500,500,500)
# sample_descriptor = "SilKet" #"AH" #"AL" #"BH" #"BL"
# imagesize = (946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #(926,925,854)
# datatype = 'float32'
# sample_descriptor = "menke_2017_est"
# imagesize =(998,998,800)
# sample_descriptor = "alkhulafi_silurian"
# imagesize = (946, 946, 390)
# datatype = 'float16'

load_structure = False
#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')

#load data
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
sample_descriptor = df['Sample_Name']

for sample in sample_descriptor:
    fig,ax = plt.subplots()
    #define extension type
    if sample == 'AH' or sample == 'AL' or sample == 'BL' or sample == 'BH':
        ext = '.raw'
        datatype = 'float32'
    else:
        ext = '.txt'    
        datatype = 'float16'
    #data file location
    vel_magnitude_file = directory + "/" + sample + "_velocity_magnitude" + ext

    #load images
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #load structure - get into loop form
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape((650,650,650))
        structure_file = directory + "/" + sample + "_structure.dat"
        structure = np.loadtxt(structure_file) #takes a long time, but using np.fromfile does not work!!!, may be better to save in different format like ASCII?
        structure = structure.reshape((650,650,650))
        #remove grains
        vel_magnitude = vel_magnitude[structure != 1]
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    #normalizing velocity field by mean
    mean = np.mean(vel_magnitude)
    vel_magnitude /= mean

    #make histogram
    bins = 10 ** np.linspace(np.log10(np.min(vel_magnitude[vel_magnitude != 0])), np.log10(np.max(vel_magnitude)),num=15)
    n,bins = np.histogram(vel_magnitude,density=True,bins = bins) 
    densities = n*np.diff(bins)
    #plot pdf
    ax.plot(bins[:-1],densities, label = sample, linewidth = 2)

    ax.semilogx()
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel('V/<V>', fontsize=15)
    ax.set_ylabel('PDF',fontsize=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(sample+"_velocityPDF.png")
    plt.close()
# plt.show()