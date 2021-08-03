from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import os
import numpy as np
import pandas as pd
import time

#customize these for each sample
# sample_descriptor = "menke_ketton"
# imagesize =(922,902,911)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
# sample_descriptor = "beadpack"
# imagesize =(500,500,500)
# sample_descriptor = "BL" #"SilKet" #"AH" #"AL" #"BH" #
# imagesize = (926,925,854) #(946,946,822) #(914,905,834) #(909,910,831) #(926,916,799) #
# sample_descriptor = "menke_2017_est"
# imagesize =(998,998,800)
sample_descriptor = "alkhulafi_silurian"
imagesize = (946, 946, 390)
datatype = 'uint8'

tic = time.perf_counter()

#data directory
directory = os.path.normpath(r'E:\FlowHet_RxnDist')
#data file location
file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

#load image
npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
npimg = npimg.reshape(imagesize)
npimg[npimg != 3] = 0

#get velocity region into labeled form
labels_out = label(npimg)

#extract region's coordinates (to get surface area) and total volume (lib writes as area)
props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
props = pd.DataFrame(props)

#convert to mesh format
verts, faces, normals, values = marching_cubes(labels_out)

#find surface area
surfacearea = mesh_surface_area(verts, faces)

#find total volume
volume = sum(props['area'])

#find and print mixing metric
mixing_metric = (surfacearea)**(1/2)/(volume)**(1/3)
print(mixing_metric)

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
