from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import os
import numpy as np
import pandas as pd
import time

#customize these for each sample
# sample_descriptor = "beadpack"
# imagesize = (500,500,500)
# sample_descriptor = "estaillades"
# imagesize =(650,650,650)
# sample_descriptor = "Ketton_10003"
# imagesize = (1000,1000,1000)
# sample_descriptor = "menke_ketton"
# imagesize =(922,902,911)
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
sample_descriptor = "menke_2017_ketton_3.6"
imagesize =(499,499,450)

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
print(sample_descriptor,mixing_metric)

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
