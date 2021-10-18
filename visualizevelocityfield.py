#import  libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d
from visual_utils import array_to_dataframe, array_to_RGB, numpy_to_ply, convert_to_structure, only_percolating_path, add_edges, high_velocity_region, make_movie
import h5py
import time
import matplotlib.pyplot as plt

#select what you want to image
plot_velocity_field = False
plot_structure = False
plot_velocity_regions = False
plot_high_velocity = True
plot_percolating_path = True

#customize the following for each sample
# sample_descriptor = "estaillades"
#limit is ~650 voxels^3
# imagesize =(650,650,650)

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)

sample_descriptor = "estaillades"

imagesize = (df.loc[sample_descriptor,'nx'],df.loc[sample_descriptor,'ny'],df.loc[sample_descriptor,'nz'])

#datatype is uint8 for velocity regions and float16 for velocity magnitude
datatype = 'uint8'

tic = time.perf_counter()

#data directory
directory = os.path.normpath(r'F:\FlowHet_RxnDist')

if plot_velocity_field == True:
    #data file location
    file_location = directory + "/" + sample_descriptor + "_velocity_magnitude.raw"
    datatype = 'float32'
    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    #get image data in correct format
    coords,color= array_to_dataframe(npimg.astype('float32'),sliceby=1)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_vel_mag.ply"
    pcd = numpy_to_ply(coords,color,file_name=filename, overwrite=True, colormap="turbo")
    pcd = pcd.uniform_down_sample(every_k_points=10)
    o3d.visualization.draw_geometries([pcd])

if plot_structure == True:
    #data file location
    file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"
    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    #get image data in correct format
    coords_bin,color_bin = array_to_dataframe(npimg.astype('float32'),sliceby=5)
    #convert velocity field into structure only (binarize) and remove non-grain voxels
    coords_bin,color_bin = convert_to_structure(coords_bin,color_bin)
    filename = directory + "/" + sample_descriptor + "_structure.ply"
    # create/load .ply file and visualize
    pcd = numpy_to_ply(coords_bin,color_bin,file_name=filename, overwrite=True, colormap="binary")
    o3d.visualization.draw_geometries([pcd])

if plot_velocity_regions == True:
    #data file location
    file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"
    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    #get image data in correct format
    coords,color = array_to_dataframe(npimg.astype('float32'),sliceby=5)
    coords,color = convert_to_structure(coords,color,show_pore_space=True)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_velocity_region.ply"
    pcd = numpy_to_ply(coords, color, file_name=filename, overwrite=True, colormap="segmented")
    o3d.visualization.draw_geometries([pcd])

if plot_high_velocity == True:
    #data file location
    file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"
    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    npimg = add_edges(npimg)
    #get image data in correct format
    coords,color = array_to_dataframe(npimg.astype('float32'),sliceby=1)
    #remove everything except for high velocity region (and edges)
    coords,color = high_velocity_region(coords,color)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_highvelocity.ply"
    pcd = numpy_to_ply(coords, color, file_name=filename, overwrite=False, colormap="segmented")
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.85,
                                    front=[ 0.0, 1.0, 0.0 ],
                                    lookat=[ npimg.shape[2]/2, npimg.shape[2]/2, npimg.shape[2]/2 ],
                                    up=[ 0.0, 0.0, 1.0 ])
    # make_movie(pcd)

if plot_percolating_path == True:
    #data file location
    file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"
    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    npimg = add_edges(npimg)
    #get image data in correct format
    coords,color = array_to_dataframe(npimg.astype('float32'),sliceby=1)
    #remove everything except for percolating path (and edges)
    coords,color = only_percolating_path(coords,color)
    # create/load .ply file and visualize
    filename = directory + "/" + sample_descriptor + "_percolating_path.ply"
    pcd = numpy_to_ply(coords, color, file_name=filename, overwrite=False, colormap="segmented")
    o3d.visualization.draw_geometries([pcd],
                                        zoom=0.85,
                                        front=[ 0.0, 1.0, 0.0 ],
                                        lookat=[ npimg.shape[2]/2, npimg.shape[2]/2, npimg.shape[2]/2 ],
                                        up=[ 0.0, 0.0, 1.0 ])
    # make_movie(pcd)

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
