from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area, perimeter
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

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
datatype = 'uint8'

tic = time.perf_counter()

#data directory
directory = os.path.normpath(r'F:\FlowHet_RxnDist')
# samples = os.listdir(directory2)

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)

pt = []
mm = []
for sample_descriptor in df.index:
    if sample_descriptor != 'menke_ketton':continue
    #structure file location
    file_location = directory + "/" + sample_descriptor + "_structure.raw"

    #load image
    struct = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    imagesize = (df.loc[sample_descriptor,'nx'],df.loc[sample_descriptor,'ny'],df.loc[sample_descriptor,'nz'])
    struct = struct.reshape(imagesize)
    # if sample_descriptor=='menke_ketton':
    #     labels_out=label(np.invert(struct),background=254)
    # elif sample_descriptor=='AH':
    #     void_volume = len(struct[struct==1])
    #     # struct[struct==1] = 0
    #     struct[struct==2] = 0
    #     struct[struct==3] = 0
    #     #convert to mesh format
    #     verts, faces, normals, values = marching_cubes(struct)

    #     #find surface area
    #     surfacearea_grains = mesh_surface_area(verts, faces)
    #     labels_out = label(struct)
    # else:
    labels_out = label(struct)

    #extract region's total volume (lib writes as area)
    # props = regionprops_table(labels_out,properties =['area'])
    # props = pd.DataFrame(props)

    #convert to mesh format
    verts, faces, normals, values = marching_cubes(labels_out,level=0)

    #find surface area
    surfacearea_grains = mesh_surface_area(verts, faces)

    #find total volume
    void_volume=len(labels_out[labels_out==0])
    # void_volume = (imagesize[0]*imagesize[1]*imagesize[2]) - sum(props['area'])

    #velocity region file location
    file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

    #load image
    npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    npimg[npimg != 3] = 0 #3 for 3d
    labels_out = label(npimg)

    #extract region's coordinates (to get surface area) and total volume (lib writes as area)
    # props = regionprops_table(labels_out,properties =['area'])
    # props = pd.DataFrame(props)

    #convert to mesh format
    verts, faces, normals, values = marching_cubes(labels_out,level=0)

    #find surface area
    surfacearea_perc = mesh_surface_area(verts, faces)

    #find total volume
    volume_perc = len(labels_out[labels_out==1])
    # volume_perc = sum(props['area'])

    mixing_metric = (surfacearea_perc/surfacearea_grains)/(volume_perc/void_volume)
    
    pc = df.loc[sample_descriptor,'pc']
    print(sample_descriptor,pc,mixing_metric)

# fig, ax = plt.subplots()
# ax.scatter(pt,mm,c='mediumpurple',edgecolors='rebeccapurple')
# ax.set_xlabel('Percolation Threshold',fontsize=15)
# ax.set_ylabel('Specific Surface Area', fontsize=15)
# ax.tick_params(labelsize=13)
# fig.tight_layout()
# plt.show()

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
