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
df = pd.read_csv("flow_transport_rxn_properties.csv",header=0)
sample_descriptor = df['Sample_Name']
# sample_descriptor = ["Ketton_10003","menke_ketton","menke_2017_est","menke_2017_ketton","menke_2017_portland","menke_2017_ketton_3.6"]
# imagesize = [(1000,1000,1000),(922,902,911),(998,998,800),(498,498,324),(800,800,800),(499,499,450)]
# sample_descriptor = ["beadpack","estaillades"]
# imagesize = [(500,500,500),(650,650,650)]
#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')
for index,sample in enumerate(sample_descriptor):
    # fig, ax = plt.subplots()

    #data file location
    vel_magnitude_file = directory + "/" + sample + "_velocity_magnitude.raw"

    #load velocity magnitude
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #remove structure
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(df['imagesize'][index])
        structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape(df['imagesize'][index])
        #remove grains
        vel_magnitude = vel_magnitude[structure != 1]
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    #normalizing velocity field by mean
    # mean = np.mean(vel_magnitude)
    # std = np.std(vel_magnitude)
    # print(sample,mean,std)
    mean = df['mu_v'][index]
    vel_magnitude /= mean
    mean = np.mean(vel_magnitude)
    std = np.std(vel_magnitude)
    # print(sample,mean,std)

    pdf,velmag_bins = np.histogram(vel_magnitude,density=True,bins = 1000) 
    cumulsum = np.cumsum(pdf)
    velmag_cdf = cumulsum/cumulsum[-1]
    
    #generate homogeneous (Gaussian) distribution
    normal = np.random.normal(loc=mean,scale = mean, size=vel_magnitude.size)
    pdf,norm_bins = np.histogram(normal,density=True,bins = 1000) 
    cumulsum = np.cumsum(pdf)
    normal_cdf = cumulsum/cumulsum[-1]

    #calculate statistical distance between velocity distribution and normal distribution
    # distance = stats.wasserstein_distance(vel_magnitude,normal)
    # print(sample,distance)

    distance_norm = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(norm_bins[:-1],normal_cdf))
    print(sample, ' distance: ', distance_norm)

