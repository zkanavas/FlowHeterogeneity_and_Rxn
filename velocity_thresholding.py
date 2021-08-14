import os
import numpy as np
# from visual_utils import array_to_dataframe, convert_to_structure, numpy_to_ply
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import time

tic = time.perf_counter()

#customize these for each sample
# sample_descriptor = "menke_ketton"
# imagesize =(922,902,911)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
# sample_descriptor = "SilKet" #"AH" #"AL" #"BH" #"BL"
# imagesize = (946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #(926,925,854)
# datatype = 'float32'
# sample_descriptor = "menke_2017_est"
# imagesize =(998,998,800)
# sample_descriptor = "alkhulafi_silurian"
# imagesize = (946, 946, 390)
sample_descriptor = "beadpack"
imagesize = (500,500,500)
datatype = 'float16'

#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')

#data file location
vel_magnitude_file = directory + "/" + sample_descriptor + "_velocity_magnitude.txt"

#load images
vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
vel_magnitude = vel_magnitude.reshape(imagesize)

mean = np.mean(vel_magnitude[vel_magnitude != 0])
vel_normalized = np.divide(vel_magnitude,mean)
percolation_threshold = np.max(vel_normalized)#*0.02
# percolation_threshold = 2.07

# continuous = False
# while not continuous:
#     percolation_threshold -= 0.01
#     print(percolation_threshold)
#     [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
#     vel_norm[x_2,y_2,z_2] = 1

#     labels_out = label(vel_norm)
#     props = regionprops_table(labels_out,properties =['label','bbox'])
#     props = pd.DataFrame(props)
#     checking_bounds = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
#     if any(checking_bounds):
#         print('done')
#         props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
#         props = pd.DataFrame(props)
#         id_box = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
#         coords = props['coords'][id_box.index[np.where(id_box)]].tolist()
#         continuous = True
# #id velocity zones
# vel_norm[x_2,y_2,z_2] = 2 #disconnected high velocity region
# vel_norm[coords[0][:,0],coords[0][:,1],coords[0][:,2]] = 3 #percolating path
# [x_1,y_1,z_1] = np.where(np.logical_and(vel_normalized < percolation_threshold, vel_normalized > 0))
# vel_norm[x_1,y_1,z_1] = 1 #stagnant zone
# #the leftovers are 0 and correspond to solid
# #save thresholded velocity field
# vel_norm.astype('uint8').tofile(directory +"/" + sample_descriptor + '_velocity_regions.txt')

def checkbounds(vel_normalized,percolation_threshold):
    vel_norm = np.zeros(vel_magnitude.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox'])
    props = pd.DataFrame(props)
    checking_bounds = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    return any(checking_bounds)

def save(vel_normalized,percolation_threshold):
    vel_norm = np.zeros(vel_magnitude.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
    props = pd.DataFrame(props)
    id_box = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    coords = props['coords'][id_box.index[np.where(id_box)]].tolist()
    #id velocity zones
    vel_norm[x_2,y_2,z_2] = 2 #disconnected high velocity region
    vel_norm[coords[0][:,0],coords[0][:,1],coords[0][:,2]] = 3 #percolating path
    [x_1,y_1,z_1] = np.where(np.logical_and(vel_normalized < percolation_threshold, vel_normalized > 0))
    vel_norm[x_1,y_1,z_1] = 1 #stagnant zone
    #the leftovers are 0 and correspond to solid

    #save thresholded velocity field
    vel_norm.astype('uint8').tofile(directory +"/" + sample_descriptor + '_velocity_regions.txt')

upper_pt = percolation_threshold
lower_pt = upper_pt/2
stop = False
while not stop:
    if not checkbounds(vel_normalized,lower_pt):
        upper_pt = lower_pt
        lower_pt = upper_pt/2
    else:
        if upper_pt - lower_pt < 1e-5:
            print(lower_pt)
            save(vel_normalized,lower_pt)
            stop=True
        else: lower_pt += lower_pt/2

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )