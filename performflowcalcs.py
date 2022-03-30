from flowfieldcomputations import convert_components_into_magnitude,earth_movers_distance,percolation_threshold,dimensionless_volume_interfacialsurfacearea_SSA
# required libraries: numpy, pandas, matplotlib, mat73, scipy, sklearn, skimage
import os
import numpy as np
import pandas as pd
import time

calc_components = False
calc_percolation_threshold = False
calc_EMD = True
calc_V_S = False

df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)

rootdir =r"H:\FlowHet_RxnDist"
folders_to_look_thru = ["Menke2015","Menke2017","AlKhulaifi2018","AlKhulaifi2019","Hinz2019","PereiraNunes2016"]
# folders_to_look_thru = ["AlKhulaifi2019"]
reaction_phase = ["final","initial"]

if calc_components:
    for (root,dirs,files) in os.walk(rootdir):
        if any(folder in root for folder in folders_to_look_thru) and any(phase in root for phase in reaction_phase):
            sample = [x for x in df.index if x in root]
            imagesize = (int(df['nx'][sample].values[0]),int(df['ny'][sample].values[0]),int(df['nz'][sample].values[0]))
            phase = [phase for phase in reaction_phase if phase in root]
            print("converting to vel mag for ", sample, phase)
            for file in files:
                if ".mat" in file:
                    #file is the matlab file of the velocity components, need to make it magnitude
                    vel_components_file = root+"/"+file
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    Ux = []
                    Uy = []
                    Uz = []
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=True,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')
                elif "Ux" in file and ".raw" in file:
                    vel_components_file = []
                    Ux = root + "/" + "Ux.raw"
                    Uy = root + "/" + "Uy.raw"
                    Uz = root +"/" + "Uz.raw"
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=False,loadfrommat=False,loadfromraw=True,loadfromdat=False,datatype = 'float64')
                elif "Ux" in file and ".dat" in file:
                    vel_components_file = []
                    Ux = root + "/" + "Ux.dat"
                    Uy = root + "/" + "Uy.dat"
                    Uz = root +"/" + "Uz.dat"
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=False,loadfrommat=False,loadfromraw=False,loadfromdat=True,datatype = 'float64')

indeces = [(ind+"_"+phase) for ind in df.index for phase in reaction_phase]
res = pd.DataFrame(columns=["manual-log","manual-log_time","manual-linear","manual-linear_time","built-in","built-in_time"],index=indeces)

for (root,dirs,files) in os.walk(rootdir):
    if any(folder in root for folder in folders_to_look_thru) and any(phase in root for phase in reaction_phase):
        sample = [x for x in df.index if x in root]
        imagesize = (int(df['nx'][sample].values[0]),int(df['ny'][sample].values[0]),int(df['nz'][sample].values[0]))
        phase = [phase for phase in reaction_phase if phase in root]
        for file in files:
            if "vel_magnitude.raw" in file:
                #file is the matlab file of the velocity components, need to make it magnitude
                vel_magnitude_file = root+"/"+file
                structure_file = []
                if calc_EMD:
                    indi = sample[0] + "_" + phase[0]
                    print("calculating EMD for ", sample,phase)

                    tic = time.time()
                    distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=False,normalize_velocity_field=True,load_structure = False,plot = False,datatype = 'float32')
                    toc = time.time()
                    res['built-in'][indi] = distance
                    res['built-in_time'][indi] = toc - tic

                    tic = time.time()
                    distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=True,normalize_velocity_field=True,load_structure = False,plot = False,datatype = 'float32',logspacing=True)
                    toc = time.time()
                    res['manual-log'][indi] = distance
                    res['manual-log_time'][indi] = toc - tic
                    
                    tic = time.time()
                    distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=True,normalize_velocity_field=True,load_structure = False,plot = False,datatype = 'float32',logspacing=False)
                    toc = time.time()
                    res['manual-linear'][indi] = distance
                    res['manual-linear_time'][indi] = toc - tic

                else: distance = []
                velocity_regions_file = root +"/_vel_regions.raw"
                if calc_percolation_threshold:
                    print("finding percolation threshold for ", sample,phase)
                    pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-2])
                else: pc = []
                if calc_V_S:
                    print("calculating V S SSA ", sample,phase)
                    volume, surfacearea, ssa = dimensionless_volume_interfacialsurfacearea_SSA(velocity_regions_file,imagesize,include_disconnected_hvr=True,datatype = 'uint8')
                else: 
                    volume = [] 
                    surfacearea = [] 
                    ssa = []
                # print(sample,phase," EMD: ",distance, " pc: ",pc, " V: ",volume, " S: ", surfacearea, " SSA: ",ssa)
res.to_csv("EMD_variations_comparison.csv")
