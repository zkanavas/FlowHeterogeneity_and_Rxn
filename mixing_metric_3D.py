from random import sample
from xml.etree.ElementTree import TreeBuilder
from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area, perimeter
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.stats as stats

calc_metric = True
calc_SSA = False
plot_metric = False
plot_SSA = False

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
directory = os.path.normpath(r'D:\FlowHet_RxnDist')
# samples = os.listdir(directory2)

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)

# samples = ["Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]
samples = ["geometry2600"]

if calc_metric == True:
    for sample_descriptor in samples:#df.index:# samples:
        # if sample_descriptor != 'Sil_HetA_High_Scan1':continue#any([sample_descriptor=='Sil_HetA_High_Scan1',sample_descriptor=='Sil_HetA_Low_Scan1',sample_descriptor=='Sil_HetB_High_Scan1',sample_descriptor=='Sil_HetB_Low_Scan1']):continue
        
        # imagesize = (df.loc[sample_descriptor,'nx'],df.loc[sample_descriptor,'ny'],df.loc[sample_descriptor,'nz'])
        imagesize=(400,400,400)
        #velocity region file location
        file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

        #load image
        npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
        npimg = npimg.reshape(imagesize)
        npimg[npimg != 0] = 1 #all pore space == 1
        labels_out = label(npimg)
        #extract region's coordinates (to get surface area) and total volume (lib writes as area)
        props = regionprops_table(labels_out,properties =['area'])
        props = pd.DataFrame(props)

        #convert to mesh format
        verts, faces, normals, values = marching_cubes(labels_out,level=0)

        #find surface area
        surfacearea_grains = mesh_surface_area(verts, faces)
        #find total volume
        void_volume = len(labels_out[labels_out==1])

        #velocity region file location
        file_location = directory + "/" + sample_descriptor + "_velocity_regions.txt"

        #load image
        npimg = np.fromfile(file_location, dtype=np.dtype(datatype)) 
        npimg = npimg.reshape(imagesize)
        npimg[npimg == 2] = 3 #includes disconnected high velocity region
        npimg[npimg != 3] = 0 #3 for 3d
        labels_out = label(npimg)

        #convert to mesh format
        verts, faces, normals, values = marching_cubes(labels_out,level=0)

        #find surface area
        surfacearea_perc = mesh_surface_area(verts, faces)

        #find total volume
        volume_perc = len(npimg[npimg!=0])

        surfacearea = (surfacearea_perc/surfacearea_grains)
        volume = (volume_perc/void_volume)


        # pc = df.loc[sample_descriptor,'pc']
        pc = 4
        print(sample_descriptor,pc,surfacearea,volume)

if calc_SSA:
    for sample_descriptor in df.index:# samples:
        print(sample_descriptor,'percolating path',round(df.loc[sample_descriptor,'SA_perc']/df.loc[sample_descriptor,'Vol_perc'],3))# == df.loc[sample_descriptor,'SSA_perc'])
        print(sample_descriptor,'high velovity',round(df.loc[sample_descriptor,'SA_hv']/df.loc[sample_descriptor,'Vol_hv'],3))# == df.loc[sample_descriptor,'SSA'])

surfacearea_ = df.SA_perc
volume_ = df.Vol_perc
ylimit = (0.006,0.075) #(0.01,0.25)
spacing = (ylimit[1]-ylimit[0])*(1/16)
print(spacing)
region = 'Percolating Path'

r_surface,p_surface = stats.spearmanr(df.pc,surfacearea_)

r_volume,p_volume = stats.spearmanr(df.pc,volume_)

# np.savetxt("surface_volume_percolating_pathway.csv",np.column_stack((pt,surfaces,volumes)),delimiter=",")
if plot_metric == True:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(df.pc,surfacearea_,c='mediumpurple',edgecolors='rebeccapurple')
    ax1.set_ylim(ylimit)
    ax1.set_xlabel('Percolation Threshold',fontsize=15)
    ax1.set_ylabel('Surface Area of '+region+'/Void', fontsize=15,color='rebeccapurple')
    ax1.tick_params(labelsize=13)
    ax1.tick_params('y',color='rebeccapurple',labelcolor='rebeccapurple')
    ax2.scatter(df.pc,volume_,c='goldenrod',edgecolors='darkgoldenrod',alpha=0.5)
    ax2.set_ylim(ylimit)
    ax2.set_ylabel('Volume of ' + region +'/Void', fontsize=15,color='darkgoldenrod')
    ax2.tick_params(labelsize=13)
    ax2.tick_params('y',color='darkgoldenrod',labelcolor='darkgoldenrod')
    string_1 = 'Spearman r: ', str(round(r_surface,2)), ' p-value: ',str(round(p_surface,2))
    string_2 = 'Spearman r: ', str(round(r_volume,2)), ' p-value: ',str(round(p_volume,2))
    ax1.text(4,np.max(surfacearea_),''.join(string_1),color ='rebeccapurple',fontsize=13)
    ax1.text(4,np.max(surfacearea_)-spacing,''.join(string_2),color='darkgoldenrod',fontsize=13)

    fig.tight_layout()
    plt.show()

specificsurfacearea = df.SSA_perc #df.SSA_perc
# ylim_ = 
region = 'Percolating Path'
r,p = stats.spearmanr(df.pc,specificsurfacearea)

if plot_SSA == True:
    fig, ax1 = plt.subplots()
    ax1.scatter(df.pc,specificsurfacearea)
    # ax1.set_ylim(ylimit)
    ax1.set_xlabel('Percolation Threshold',fontsize=15)
    ax1.set_ylabel('SSA of '+region+'/Void', fontsize=15)
    ax1.tick_params(labelsize=13)
    string_1 = 'Spearman r: ', str(round(r,2)), ' p-value: ',str(round(p,2))
    ax1.text(4,np.max(specificsurfacearea),''.join(string_1),fontsize=13)
    
    fig.tight_layout()
    plt.show()

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
