from scipy import stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

#customize these for each sample
# sample_descriptor = "SilKet" #"AH" #"AL" #"BH" #"BL"
# sample_descriptor = ["SilKet","AH","AL","BH","BL"]
# imagesize = (946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #(926,925,854)
datatype = 'float32'
load_structure = False
plot = False
greaterthan0 = False
Gaussian = False
Lognormal = True
df = pd.read_csv("flow_transport_rxn_properties.csv",header=0)
sample_descriptor = df['Sample_Name']
# sample_descriptor = ["fracturedB"]
# imagesize = [(300,300,400)]

# sample_descriptor = "Sil_HetA_High_Scan1"
# imagesize =(839,849,812)
# sample_descriptor = ["beadpack","estaillades"]
# imagesize = [(500,500,500),(650,650,650)]

# sample_descriptor = ["Sil_HetA_High_Scan1","Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]
# imagesize = [(936,936,787),(911,914,829),(903,889,785)]

#data directory
directory = os.path.normpath(r'D:\FlowHet_RxnDist')
for index,sample in enumerate(sample_descriptor):
    # if sample != "beadpack" and sample != "AH": continue
    
    #data file location
    vel_magnitude_file = directory + "/" + sample + "_velocity_magnitude.raw"

    #load velocity magnitude
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #remove structure
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(df['imagesize'][index])
        structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape((df['nx'][index],df['ny'][index],df['nz'][index]))
        #remove grains
        vel_magnitude = vel_magnitude[structure != 1]
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    #normalizing velocity field by mean
    mean = np.mean(vel_magnitude)

    vel_magnitude /= mean
    mean = np.mean(vel_magnitude)
    std = np.std(vel_magnitude)
    print(sample,mean,std)

    pdf,velmag_bins = np.histogram(vel_magnitude,density=True,bins = 1000) 
    cumulsum = np.cumsum(pdf)
    velmag_cdf = cumulsum/cumulsum[-1]
    if Gaussian == True:
        #generate homogeneous (Gaussian) distribution
        generated = np.random.normal(loc=mean,scale = std, size=vel_magnitude.size)
        pdf,gen_bins = np.histogram(generated,density=True,bins = 1000) 
        cumulsum = np.cumsum(pdf)
        gen_cdf = cumulsum/cumulsum[-1]
    elif Lognormal == True:
        #generate homogeneous (log-normal) distribution
        generated = np.random.lognormal(mean=mean,sigma = std, size=vel_magnitude.size)
        pdf,gen_bins = np.histogram(generated,density=True,bins = 1000) 
        cumulsum = np.cumsum(pdf)
        gen_cdf = cumulsum/cumulsum[-1]


    if gen_bins[-2] < velmag_bins[-2]:
        gen_bins = np.append(gen_bins,[velmag_bins[-2],velmag_bins[-1]]) 
        gen_cdf = np.append(gen_cdf,[1,1])
    else:
        velmag_bins = np.append(velmag_bins,[gen_bins[-2],gen_bins[-1]])
        velmag_cdf = np.append(velmag_cdf,[1,1])
    if Lognormal == True:
        if gen_bins[0] < velmag_bins[0]:
            velmag_cdf = np.insert(velmag_cdf,0,0)
            velmag_bins = np.insert(velmag_bins, 0,np.min(gen_bins)) 
        else:
            gen_cdf = np.insert(gen_cdf,0,0)
            gen_bins = np.insert(gen_bins, 0,np.min(velmag_bins)) 

    if Gaussian == True:
        velmag_cdf = np.insert(velmag_cdf,0,0)
        velmag_bins = np.insert(velmag_bins, 0,0)
        if not greaterthan0:
            velmag_cdf = np.insert(velmag_cdf,0,0)
            velmag_bins = np.insert(velmag_bins, 0,np.min(gen_bins)) 
        else: 
            gen_cdf = np.insert(gen_cdf,0,0)
            gen_bins = np.insert(gen_bins, 0,0)
            real_space = gen_bins[:-1]>=0

    #calculate statistical distance between velocity distribution and normal distribution
    distance = stats.wasserstein_distance(vel_magnitude,generated)
    print(sample," wasserstein: ",distance)
    
    # auc_obs = metrics.auc(velmag_bins[:-1],velmag_cdf)
    # auc_norm_truncated = metrics.auc(norm_bins[:-1][real_space],normal_cdf[real_space])
    # auc_norm_all = metrics.auc(norm_bins[:-1],normal_cdf)
    if greaterthan0:
        distance_norm = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(gen_bins[:-1][real_space],gen_cdf[real_space]))
    else:
        distance_norm = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(gen_bins[:-1],gen_cdf))

    print(sample, ' distance: ', distance_norm)
    if plot == True:
        fig,ax = plt.subplots()
        ax.set_title(sample)
        ax.plot(velmag_bins[:-1],velmag_cdf,label='true distribution')
        if not greaterthan0:
            ax.plot(gen_bins[:-1],gen_cdf,label='log-normal distribution')
        else:
            ax.plot(gen_bins[:-1][real_space],gen_cdf[real_space],label='log-normal distribution')
        ax.semilogx()
        ax.tick_params(axis='both',labelsize=14)
        ax.set_xlabel('V/<V>', fontsize=15)
        ax.set_ylabel('CDF',fontsize=15)
        ax.legend()
        fig.tight_layout()
if plot == True: plt.show()
