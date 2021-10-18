from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area, perimeter
from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.stats as stats

collect_data = False
plot = True

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
# directory = os.path.normpath(r'F:\Big_Samples_001')#(r'F:\FlowHet_RxnDist')
if collect_data == True:
    directory = os.path.normpath(r'C:\Users\zkana\Box\Morales Lab\Z_Kanavas\Differential_Evolution_Project\01_COMSOL_automation\03_Images\2_PT_MatlabData\Sample_A')
    directory2 = os.path.normpath(r'C:\Users\zkana\Box\Morales Lab\Z_Kanavas\Differential_Evolution_Project\MastersWork\Big_Samples_001\Sample_A')
    directory3 = os.path.normpath(r'C:\Users\zkana\Box\Morales Lab\Z_Kanavas\Differential_Evolution_Project\MastersWork')
    # samples = os.listdir(directory2)

    percthres = pd.read_csv(directory3+'/PercolationThresholds.txt',names=['Sample_Names','Percolation_Threshold'],sep='\t')

    filetype = "0000.txt" # "_velocity_regions.txt"
    pt = []
    surfaces = []
    volumes = []

    for sample_descriptor_ in percthres.Sample_Names.values:
    # for sample_descriptor in samples:
        sample_descriptor = sample_descriptor_.replace("'","")
        # if sample_descriptor != 'A417E01_' and sample_descriptor != 'A127E01_':continue
        #data file location
        file_location = directory2 + "/" + sample_descriptor + "/" + sample_descriptor + filetype
        if sample_descriptor.endswith('E01_'): 
            subfolder = 'Eroding_01'
        # elif sample_descriptor.endswith('E03_'):
        #     subfolder = 'Eroding_03'
        # elif sample_descriptor.endswith('E05_'): 
        #     subfolder = 'Eroding_05'
        # elif sample_descriptor.endswith('E07_'): 
        #     subfolder = 'Eroding_07'
        # elif sample_descriptor.endswith('E09_'): 
        #     subfolder = 'Eroding_09'
        else: continue

        #load image
        # struct = np.fromfile(file_location)#, dtype=np.dtype(datatype)) 
        struct = np.loadtxt(file_location,dtype=np.dtype(datatype))
        struct[struct == 255] = 1
        # npimg = npimg.reshape(imagesize)
        # npimg[npimg != 3] = 0

        perimeter_ = perimeter(struct,neighbourhood=8)
        area = len(np.where(struct==1)[0])

        npimg = loadmat(directory +'/'+ subfolder +'/' +sample_descriptor + '.mat')['plotting_final']
        # npimg[npimg == 3] = 4 #includes disconnected high velocity region
        npimg[npimg != 4] = 0 #3 for 3d
        npimg[npimg == 4] = 1
        sa = perimeter(npimg,neighbourhood=8)
        vol = len(np.where(npimg==1)[0])
        # mixing_metric = (sa/perimeter_)/(vol/area)
        surfacearea = sa/perimeter_
        volume = vol/area
        #get velocity region into labeled form
        # labels_out = label(npimg)

        # #extract region's coordinates (to get surface area) and total volume (lib writes as area)
        # props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
        # props = pd.DataFrame(props)


        
        # #convert to mesh format
        # verts, faces, normals, values = marching_cubes(labels_out)

        # #find surface area
        # surfacearea = mesh_surface_area(verts, faces)

        # #find total volume
        # volume = sum(props['area'])

        # #find and print mixing metric
        # mixing_metric = (surfacearea/perimeter_)/(volume/area)
        pc = percthres.Percolation_Threshold[np.where(percthres.Sample_Names==sample_descriptor_)[0]].values
        pt.append(pc[0])
        surfaces.append(surfacearea)
        volumes.append(volume)
        print(sample_descriptor,pc,surfacearea,volume)

    np.savetxt("surface_volume_percolating_pathway.csv",np.column_stack((pt,surfaces,volumes)),delimiter=",")

df = pd.read_csv('surface_volume_percolating_pathway.csv',names=['Percolation_Threshold','Surface_Area','Volume'])
r_surface,p_surface = stats.spearmanr(df.Percolation_Threshold,df.Surface_Area)

r_volume,p_volume = stats.spearmanr(df.Percolation_Threshold,df.Volume)

if plot == True:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(df.Percolation_Threshold,df.Surface_Area,c='mediumpurple',edgecolors='rebeccapurple',alpha=0.75)
    ax1.set_ylim(0.05,0.7)
    ax1.set_xlabel('Percolation Threshold',fontsize=15)
    ax1.set_ylabel('Surface Area', fontsize=15,color='rebeccapurple')
    ax1.tick_params(labelsize=13)
    ax1.tick_params('y',color='rebeccapurple',labelcolor='rebeccapurple')
    ax2.scatter(df.Percolation_Threshold,df.Volume,c='goldenrod',edgecolors='darkgoldenrod',alpha=0.5)
    ax2.set_ylim(0.05,0.7)
    ax2.set_ylabel('Volume', fontsize=15,color='darkgoldenrod')
    ax2.tick_params(labelsize=13)
    ax2.tick_params('y',color='darkgoldenrod',labelcolor='darkgoldenrod')
    string_1 = 'Spearman r: ', str(round(r_surface,2)), ' p-value: ',str(round(p_surface,2))
    string_2 = 'Spearman r: ', str(round(r_volume,2)), ' p-value: ',str(round(p_volume,2))
    ax2.text(np.mean(df.Percolation_Threshold),np.max(df.Volume),''.join(string_1),color ='rebeccapurple',fontsize=13)
    ax2.text(np.mean(df.Percolation_Threshold),np.max(df.Volume)-0.05,''.join(string_2),color='darkgoldenrod',fontsize=13)
    fig.tight_layout()
    plt.savefig("2d_perc_SA_vol.png")
    plt.show()

toc = time.perf_counter()
print("time elapsed: " + str(toc-tic) + " seconds" )
