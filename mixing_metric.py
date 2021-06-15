from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import os
import numpy as np
import pandas as pd

#customize these for each sample
sample_descriptor = "beadpack"
imagesize =(500,500,500)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
datatype = 'uint8'

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\Data')
#data file location
file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

#load image
npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
npimg = npimg.reshape(imagesize)

#get velocity region into labeled form
labels_out = label(npimg)

#extract region's coordinates (to get surface area) and total volume (lib writes as area)
props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
props = pd.DataFrame(props)

#convert to mesh format
verts, faces, normals, values = marching_cubes(labels_out,level=1)

#find surface area
surfacearea = mesh_surface_area(verts, faces)

#find total volume
volume = sum(props['area'])

#find and print mixing metric
mixing_metric = (surfacearea)**(1/2)/(volume)**(1/3)
print(mixing_metric)