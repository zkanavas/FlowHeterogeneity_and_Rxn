import os
import numpy as np
from continuity_check import check_continuity
import networkx as nx
from visual_utils import array_to_dataframe, convert_to_structure, numpy_to_ply
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import open3d as o3d

#customize these for each sample
sample_descriptor = "beadpack"
imagesize =(500,500,500)
datatype = 'float64'
plot = False

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\Data')

#data files
Ux_velfield = directory + '/Ux_beadpack_t0.raw'
Uy_velfield = directory + '/Uy_beadpack_t0.raw'
Uz_velfield = directory + '/Uz_beadpack_t0.raw'

#load images
Ux_array = np.fromfile(Ux_velfield, dtype=np.dtype(datatype)) #x-direction
Ux_array = Ux_array.reshape(imagesize)
Uy_array = np.fromfile(Uy_velfield, dtype=np.dtype(datatype)) #y-direction
Uy_array = Uy_array.reshape(imagesize)
Uz_array = np.fromfile(Uz_velfield, dtype=np.dtype(datatype)) #z-direction
Uz_array = Uz_array.reshape(imagesize)

#calculate magnitude
vel_magnitude = sum([Ux_array**2, Uy_array**2, Uz_array**2])**(1/2)

#save magnitude file
# vel_magnitude.astype('float16').tofile(sample_descriptor + '_velocity_magnitude.txt')

mean = np.mean(vel_magnitude[vel_magnitude != 0])
vel_normalized = np.divide(vel_magnitude,mean)
percolation_threshold = np.max(vel_normalized)*0.5

vel_norm = np.zeros(vel_magnitude.shape)

continuous = False
while not continuous:
    percolation_threshold -= 0.01
    print(percolation_threshold)
    [z_2,y_2,x_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[z_2,y_2,x_2] = 1

    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','area'])
    props = pd.DataFrame(props)
    # props.sort_values(by='bbox-0',inplace=True)
    if any(props['bbox-3'][props['bbox-0']==0] == imagesize[0]):
        continuous = True

# surface_area = props['perimeter'].sum()
volume = props['area'].sum()
# mixing_metric = (surface_area)^(1/2)/(volume)^(1/3)
# print(mixing_metric)

#plot result
if plot == True:
    #get image data in correct format
    coords_bin,color_bin = array_to_dataframe(vel_norm.astype('float32'))
    #convert velocity field into structure only (binarize) and remove non-grain voxels
    coords_bin,color_bin = convert_to_structure(coords_bin,color_bin,show_pore_space = True)
    # create/load .ply file and visualize
    pcd = numpy_to_ply(coords_bin,color_bin,file_name="highspeed.ply", overwrite=True)
    o3d.visualization.draw_geometries([pcd])
