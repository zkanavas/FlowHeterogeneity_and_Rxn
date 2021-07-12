import os
import numpy as np
from visual_utils import array_to_dataframe, convert_to_structure, numpy_to_ply
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import open3d as o3d

#customize these for each sample
sample_descriptor = "menke_ketton"
imagesize =(922,902,911)
datatype = 'float16'

#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')

#data file location
vel_magnitude_file = directory + "/" + sample_descriptor + "_velocity_magnitude.txt"

#load images
vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) #x-direction
vel_magnitude = vel_magnitude.reshape(imagesize)

mean = np.mean(vel_magnitude[vel_magnitude != 0])
vel_normalized = np.divide(vel_magnitude,mean)
percolation_threshold = np.max(vel_normalized)*0.1

vel_norm = np.zeros(vel_magnitude.shape)

continuous = False
while not continuous:
    percolation_threshold -= 0.01
    print(percolation_threshold)
    [z_2,y_2,x_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[z_2,y_2,x_2] = 1

    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','bbox_area'])
    props = pd.DataFrame(props)
    if any(props['bbox-3'][props['bbox-0']==0] == imagesize[0]):
        region_label = np.where(props['bbox-3'][props['bbox-0']==0] == imagesize[0])[0]
        continuous = True

props = regionprops_table(labels_out,properties =['label','coords','bbox_area'])
props = pd.DataFrame(props)
coords = props['coords'][region_label[0]]

#id velocity zones
vel_norm[z_2,y_2,x_2] = 2 #disconnected high velocity region
vel_norm[coords[:,0],coords[:,1],coords[:,2]] = 3 #percolating path
[z_1,y_1,x_1] = np.where(np.logical_and(vel_normalized < percolation_threshold, vel_normalized > 0))
vel_norm[z_1,y_1,x_1] = 1 #stagnant zone
#the leftovers are 0 and correspond to solid

#save thresholded velocity field
vel_norm.astype('uint8').tofile(directory +"/" + sample_descriptor + '_velocity_regions.txt')
