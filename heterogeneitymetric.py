from scipy import stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#customize these for each sample
# sample_descriptor = "SilKet" #"AH" #"AL" #"BH" #"BL"
# sample_descriptor = ["SilKet","AH","AL","BH","BL"]
# imagesize = (946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #(926,925,854)
datatype = 'float32'

df = pd.read_csv("flow_transport_rxn_properties.csv",header=0)

#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')
for index,sample in enumerate(df.Sample_Name):
    fig, ax = plt.subplots()
    #define extension
    if sample == 'beadpack':# or sample == 'estaillades' or sample == 'menke_ketton':
        ext = '.txt'
    else:
        continue #ext = '.raw'    

    #data file location
    vel_magnitude_file = directory + "/" + sample + "_velocity_magnitude" + ext

    #load velocity magnitude
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    vel_magnitude = vel_magnitude[vel_magnitude != 0]
    #calculate mean velocity
    mean_velocity = np.mean(vel_magnitude)
    std_velocity = np.std(vel_magnitude)
    #generate homogeneous (Gaussian) distribution
    normal = np.random.normal(loc=mean_velocity,scale = std_velocity, size=vel_magnitude.size)

    #calculate statistical distance between velocity distribution and normal distribution
    distance = stats.wasserstein_distance(vel_magnitude,normal)

    print(sample,df.pc[index],distance)

