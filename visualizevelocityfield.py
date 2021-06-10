#import  libraries
import os
import numpy as np
import matplotlib.pyplot as plt
# from pyntcloud import PyntCloud
import pandas as pd
import open3d as o3d
from visual_utils import array_to_dataframe, array_to_RGB, numpy_to_ply, convert_to_structure
import h5py

plot_velocity_field = True
plot_structure = False

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\PracticeFolder')

Sample = directory + "/beadpack_velocity_magnitude.txt"

#load image
npimg = np.fromfile(Sample, dtype=np.dtype('float16')) #not 100% sure about the data-type, I know it is 64-bit
imagesize =(500,500,500)
npimg = npimg.reshape(imagesize)

if plot_velocity_field == True:
    #get image data in correct format
    coords,color= array_to_dataframe(npimg.astype('float16'),sliceby=1)
    coords,color = convert_to_structure(coords,color,show_pore_space=True)
    # create/load .ply file and visualize
    pcd = numpy_to_ply(coords,color,file_name="beadpack_vel_mag.ply", overwrite=True)
    pcd = pcd.uniform_down_sample(every_k_points=5)
    o3d.visualization.draw_geometries([pcd])

elif plot_structure == True:
    #get image data in correct format
    coords_bin,color_bin = array_to_dataframe(npimg.astype('float32'))
    #convert velocity field into structure only (binarize) and remove non-grain voxels
    coords_bin,color_bin = convert_to_structure(coords_bin,color_bin)
    # create/load .ply file and visualize
    pcd = numpy_to_ply(coords_bin,color_bin,file_name="structure.ply", overwrite=True)
    o3d.visualization.draw_geometries([pcd])


