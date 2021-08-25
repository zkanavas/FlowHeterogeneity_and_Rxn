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
plot = True
load_structure = True
#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')

#load data
# df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
# sample_descriptor = df['Sample_Name']

sample_descriptor = ["AL","AH","BL","BH"]
imagesize = [(909,910,831),(914,905,834),(926,925,854),(926,916,799)]
color = ['b--','b-','r--','r-']
flowrate = [.1,.5,.1,.5]
fig,ax = plt.subplots()
for count,sample in enumerate(sample_descriptor):
    # fig,ax = plt.subplots()
    #define extension type
    datatype = 'float32'
    #data file location
    vel_magnitude_file = directory + "/" + sample + "_velocity_magnitude.raw"

    #load images
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #load structure - get into loop form
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(imagesize[count])
        structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape(imagesize[count])
        #remove grains
        vel_magnitude = vel_magnitude[structure == 1]
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    #normalizing velocity field by mean
    mean = np.mean(vel_magnitude)
    darcyvelocity = (flowrate[count])/(1000**2)
    std = np.std(vel_magnitude)
    var = std**2
    print(sample,mean,var,std)
    vel_magnitude /= darcyvelocity

    #make histogram
    if plot == True:
        bins = 10 ** np.linspace(np.log10(np.min(vel_magnitude[vel_magnitude != 0])), np.log10(np.max(vel_magnitude)),num=100)
        n,bins = np.histogram(vel_magnitude,density=True,bins = bins) 
        densities = n*np.diff(bins)
        #plot pdf
        ax.plot(bins[:-1],densities, color[count],label = sample, linewidth = 2)
if plot == True:
    ax.semilogx()
    ax.tick_params(axis='both',labelsize=14)
    # ax.set_xlim(1e-10,1e5)
    ax.set_xlabel('V/VD', fontsize=15)
    ax.set_ylabel('PDF',fontsize=15)
    ax.legend()
    plt.tight_layout()
    # plt.savefig(sample+"_velocityPDF.png")
    # plt.close()
    plt.show()
# plt.savefig("velocity_distributions.png")
# plt.close()