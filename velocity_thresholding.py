import os
import numpy as np
# from visual_utils import array_to_dataframe, convert_to_structure, numpy_to_ply
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import time
import matplotlib.pyplot as plt

# tic = time.perf_counter()

# sample_descriptor = "SilKet" #"AH" #"AL" #"BH" #"BL"
# imagesize = (946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #(926,925,854)
# datatype = 'float32'

# sample_descriptor = "beadpack"
# imagesize = (500,500,500)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
# sample_descriptor = "Ketton_10003"
# imagesize = (1000,1000,1000)
# sample_descriptor = "AH"
# imagesize = (914, 905, 834)
# sample_descriptor = "AL"
# imagesize = (909, 910, 831)
# sample_descriptor = "BH"
# imagesize = (926, 916, 799)
# sample_descriptor = "BL"
# imagesize = (926,925,854)
# sample_descriptor = "menke_2017_est"
# imagesize =(998,998,800)
# sample_descriptor = "menke_2017_ketton"
# imagesize =(498,498,324)
# sample_descriptor = "menke_2017_portland"
# imagesize =(800,800,800)
# sample_descriptor = "menke_2017_ketton_3.6"
# imagesize =(499,499,450)
# sample_descriptor = "fracturedB"
# imagesize = (300,300,400)

# sample_descriptor = "Sil_HetA_High_Scan1"
# imagesize =(839,849,812)

# sample_descriptor = "Sil_HetA_Low_Scan1"
# imagesize =(936,936,787)

# sample_descriptor = "Sil_HetB_High_Scan1"
# imagesize =(911,914,829)

# sample_descriptor = "Sil_HetB_Low_Scan1"
# imagesize =(903,889,785)

def checkbounds(vel_normalized,percolation_threshold):
    vel_norm = np.zeros(vel_normalized.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','area'])
    props = pd.DataFrame(props)
    checking_bounds = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    return any(checking_bounds)

def save(vel_normalized,percolation_threshold):
    vel_norm = np.zeros(vel_normalized.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
    props = pd.DataFrame(props)
    id_box = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    # print(percolation_threshold,props['area'][id_box.index[np.where(id_box)]])
    # print("number of existing paths: ", len(np.where(id_box)[0]))
    coords = props['coords'][id_box.index[np.where(id_box)]].tolist()
    #id velocity zones
    vel_norm[x_2,y_2,z_2] = 2 #disconnected high velocity region
    vel_norm[coords[0][:,0],coords[0][:,1],coords[0][:,2]] = 3 #percolating path
    [x_1,y_1,z_1] = np.where(np.logical_and(vel_normalized < percolation_threshold, vel_normalized > 0))
    vel_norm[x_1,y_1,z_1] = 1 #stagnant zone
    #the leftovers are 0 and correspond to solid

    #save thresholded velocity field
    vel_norm.astype('uint8').tofile(directory +"/" + sample_descriptor + '_velocity_regions.txt')

# tolerance = np.logspace(-5,0,6)
tolerance = [1e-2]

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)

#data directory
directory = os.path.normpath(r'F:\FlowHet_RxnDist')
tic = time.perf_counter()

samples = ["geometry0000"]


directory = [r"F:\FlowHet_RxnDist\menke_2017_ketton_reaction_extraspace\Batch001\1DStatistics_Batch001\StokesResult",r"F:\FlowHet_RxnDist\menke_2017_ketton_reaction_extraspace\Batch100\1DStatistics_Batch100\StokesResult"]
samples = ["menke_2017_ketton_setPa_initial","menke_2017_ketton_setPa_final"]

# for sample_descriptor in df.index:
for count,sample_descriptor in enumerate(samples):
    # if sample_descriptor !="Sil_HetA_High_Scan1":continue    
    # imagesize = (df.loc[sample_descriptor,'nx'],df.loc[sample_descriptor,'ny'],df.loc[sample_descriptor,'nz'])
    imagesize = (498,498,344)
    datatype = 'float32'

    #data file location
    vel_magnitude_file = directory[count] + "/" + sample_descriptor + "_velocity_magnitude.raw"

    #bounding bbox
    # bbox_min = 'bbox-0'
    # bbox_max = 'bbox-3'

    #load images
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    vel_magnitude = vel_magnitude.reshape(imagesize)

    mean = np.mean(vel_magnitude[vel_magnitude != 0])
    vel_normalized = np.divide(vel_magnitude,mean)  
    percolation_threshold = np.max(vel_normalized)#*0.02

    upper_pt = percolation_threshold
    lower_pt = upper_pt/2
    stop = False
    while not stop:
        print(upper_pt,lower_pt)
        if not checkbounds(vel_normalized,lower_pt):
            upper_pt_new = lower_pt
            lower_pt -= (upper_pt-lower_pt)/2 
            upper_pt = upper_pt_new
        else:
            difference = upper_pt - lower_pt
            if difference < tolerance:
                print(sample_descriptor,' final pc: ',lower_pt)
                # save(vel_normalized,lower_pt)
                stop=True
            else: 
                lower_pt += (upper_pt-lower_pt)/2

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds")