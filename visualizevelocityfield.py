#import  libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from visual_utils import array_to_dataframe, array_to_RGB, numpy_to_ply, convert_to_structure, only_percolating_path
import h5py
import time

plot_velocity_field = False
plot_structure = False
plot_velocity_regions = False
plot_percolating_path = True

sample_descriptor = "estaillades"
imagesize =(650,650,650)
datatype = 'uint8'

tic = time.perf_counter()

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\Data')
#data file location
file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

#load image
npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
npimg = npimg.reshape(imagesize)


if plot_velocity_field == True:
    #get image data in correct format
    coords,color= array_to_dataframe(npimg.astype('float32'),sliceby=1)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_vel_mag.ply"
    pcd = numpy_to_ply(coords,color,file_name=filename, overwrite=False, colormap="turbo")
    pcd = pcd.uniform_down_sample(every_k_points=5)
    o3d.visualization.draw_geometries([pcd])

elif plot_structure == True:
    #get image data in correct format
    coords_bin,color_bin = array_to_dataframe(npimg.astype('float32'),sliceby=1)
    #convert velocity field into structure only (binarize) and remove non-grain voxels
    coords_bin,color_bin = convert_to_structure(coords_bin,color_bin)
    filename = directory + "/" + sample_descriptor + "_structure.ply"
    # create/load .ply file and visualize
    pcd = numpy_to_ply(coords_bin,color_bin,file_name=filename, overwrite=False, colormap="binary")
    o3d.visualization.draw_geometries([pcd])

elif plot_velocity_regions == True:
    #get image data in correct format
    coords,color = array_to_dataframe(npimg.astype('float32'),sliceby=5)
    # coords,color = only_percolating_path(coords,color)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_velocity_region.ply"
    pcd = numpy_to_ply(coords, color, file_name=filename, overwrite=True, colormap="segmented")
    o3d.visualization.draw_geometries([pcd])

elif plot_percolating_path == True:
    #get image data in correct format
    coords,color = array_to_dataframe(npimg.astype('float32'),sliceby=10)
    #remove everything except for percolating path
    coords,color = only_percolating_path(coords,color)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_percolating_path.ply"
    pcd = numpy_to_ply(coords, color, file_name=filename, overwrite=False, colormap="segmented")
    o3d.visualization.draw_geometries([pcd])

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
