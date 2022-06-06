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
load_structure = False
#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')

#load data
# df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
# sample_descriptor = df['Sample_Name']

# sample_descriptor = ["AL","AH","BL","BH"]
# imagesize = [(909,910,831),(914,905,834),(926,925,854),(926,916,799)]
# color = ['b--','b-','r--','r-']
# flowrate = [.1,.5,.1,.5]

directory = [r"F:\FlowHet_RxnDist\menke_2017_ketton_reaction_extraspace\Batch001\1DStatistics_Batch001\StokesResult",r"F:\FlowHet_RxnDist\menke_2017_ketton_reaction_extraspace\Batch100\1DStatistics_Batch100\StokesResult"]
sample_descriptor = ["menke_2017_ketton_setPa_initial","menke_2017_ketton_setPa_final"]

directory = [r"D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.6\baseline\Batch100",r"D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.6\structures\62minflowfield_pressuredrop"]
sample_descriptor = ["exp","simi"]

# directory = [r"F:\FlowHet_RxnDist\beadpack_uniform_reaction\Batch001\LIRStokesResult_Batch001",r"F:\FlowHet_RxnDist\beadpack_uniform_reaction\Batch098\LIRStokesResult_Batch098"]
# sample_descriptor = ["beadpack_uniform_initial","beadpack_uniform_final"]
labels=["simi","exp"]
color = ['r-','b-']

fig,ax = plt.subplots()
# for count,sample in enumerate(sample_descriptor):
for count,dir in enumerate(directory):
    # fig,ax = plt.subplots()
    #define extension type
    datatype = 'float32'
    #data file location
    # vel_magnitude_file = directory[count] + "/" + sample + "_velocity_magnitude.raw"
    vel_magnitude_file = dir + "/" + "vel_magnitude.raw"

    #load images
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #load structure - get into loop form
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(imagesize[count])
        structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape(imagesize[count])
        #remove grains
        total_volume = vel_magnitude.size
        vel_magnitude = vel_magnitude[structure == 1]
        porosity = vel_magnitude.size/total_volume
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    # print((sum(vel_magnitude*5.2e-6))/(1000**2))
    #normalizing velocity field by mean
    mean = np.mean(vel_magnitude)
    # darcyvelocity = mean*porosity #NO! average LINEAR velocity, must be in direction of flow (z)
    # std = np.std(vel_magnitude)
    # var = std**2
    # print(sample,mean,std)
    # vel_magnitude /= mean
    mean = np.mean(vel_magnitude)
    std = np.std(vel_magnitude)
    print(labels[count],mean,std)

    #make histogram
    if plot == True:
        bins1 = 10 ** np.linspace(np.log10(np.min(vel_magnitude[vel_magnitude != 0])), np.log10(np.max(vel_magnitude)),num=256)
        n,bins = np.histogram(vel_magnitude[vel_magnitude!=0],density=True,bins = bins1) 
        densities = n*np.diff(bins)
        #plot pdf
        # bins= np.logspace(np.log10(np.min(vel_magnitude[vel_magnitude != 0])), np.log10(np.max(vel_magnitude)),num=100)
        # ax.hist(vel_magnitude[vel_magnitude!=0],bins=bins,alpha=0.5,density=True)
        ax.plot(bins[:-1],densities, color[count],label = labels[count], linewidth = 2)
if plot == True:
    ax.semilogx()
    ax.tick_params(axis='both',labelsize=14)
    # ax.set_xlim(1e-10,1e5)
    ax.set_xlabel('V', fontsize=15)
    ax.set_ylabel('PDF',fontsize=15)
    ax.legend()
    plt.tight_layout()
    # plt.savefig(sample+"_velocityPDF.png")
    # plt.close()
    plt.show()
# plt.savefig("velocity_distributions.png")
# plt.close()