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
directory = os.path.normpath(r'F:\FlowHet_RxnDist')
for index,sample in enumerate(sample_descriptor):
    # if sample != "Sil_HetA_High_Scan1": continue
    
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
    # std = np.std(vel_magnitude)
    # print(sample,mean,std)
    # mean = df['mu_v'][index]
    # mean = df.loc[df.Sample_Name == sample].mu_v.values
    vel_magnitude /= mean
    mean = np.mean(vel_magnitude)
    std = np.std(vel_magnitude)
    print(sample,mean,std)

    pdf,velmag_bins = np.histogram(vel_magnitude,density=True,bins = 1000) 
    cumulsum = np.cumsum(pdf)
    velmag_cdf = cumulsum/cumulsum[-1]

    #generate homogeneous (Gaussian) distribution
    normal = np.random.normal(loc=mean,scale = std, size=vel_magnitude.size)
    pdf,norm_bins = np.histogram(normal,density=True,bins = 1000) 
    cumulsum = np.cumsum(pdf)
    normal_cdf = cumulsum/cumulsum[-1]

    velmag_cdf = np.insert(velmag_cdf,0,0)
    velmag_bins = np.insert(velmag_bins, 0,0)
    if norm_bins[-2] < velmag_bins[-2]:
        norm_bins = np.append(norm_bins,[velmag_bins[-2],velmag_bins[-1]]) 
        normal_cdf = np.append(normal_cdf,[1,1])
    else:
        velmag_bins = np.append(velmag_bins,[norm_bins[-2],norm_bins[-1]])
        velmag_cdf = np.append(velmag_cdf,[1,1])
    if not greaterthan0:
        velmag_cdf = np.insert(velmag_cdf,0,0)
        velmag_bins = np.insert(velmag_bins, 0,np.min(norm_bins)) 
    else: 
        normal_cdf = np.insert(normal_cdf,0,0)
        norm_bins = np.insert(norm_bins, 0,0)
        real_space = norm_bins[:-1]>=0

    #calculate statistical distance between velocity distribution and normal distribution
    # distance = stats.wasserstein_distance(vel_magnitude,normal)
    # print(sample," wasserstein: ",distance)
    
    # auc_obs = metrics.auc(velmag_bins[:-1],velmag_cdf)
    # auc_norm_truncated = metrics.auc(norm_bins[:-1][real_space],normal_cdf[real_space])
    # auc_norm_all = metrics.auc(norm_bins[:-1],normal_cdf)
    if greaterthan0:
        distance_norm = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(norm_bins[:-1][real_space],normal_cdf[real_space]))
    else:
        distance_norm = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(norm_bins[:-1],normal_cdf))

    print(sample, ' distance: ', distance_norm)
    if plot == True:
        fig,ax = plt.subplots()
        ax.set_title(sample)
        ax.plot(velmag_bins[:-1],velmag_cdf,label='true distribution')
        if not greaterthan0:
            ax.plot(norm_bins[:-1],normal_cdf,label='normal distribution')
        else:
            ax.plot(norm_bins[:-1][real_space],normal_cdf[real_space],label='normal distribution')
        ax.tick_params(axis='both',labelsize=14)
        ax.set_xlabel('V/<V>', fontsize=15)
        ax.set_ylabel('CDF',fontsize=15)
        ax.legend()
        fig.tight_layout()
if plot == True: plt.show()
