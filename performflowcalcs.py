from flowfieldcomputations import convert_components_into_magnitude,earth_movers_distance,percolation_threshold,dimensionless_volume_interfacialsurfacearea_SSA
# required libraries: numpy, pandas, matplotlib, mat73, scipy, sklearn, skimage
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import gd
from gd import stringmap
from gd import guf
import h5py
pull_out_components = True
calc_components = True
calc_percolation_threshold = True
calc_EMD = True
calc_V_S = True

calc_components_single = False

df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)

rootdir =r"H:\FlowHet_RxnDist"
folders_to_look_thru = ["Menke2017","PereiraNunes2016"]#"Menke2015","AlKhulaifi2018","AlKhulaifi2019"]#,""]
# folders_to_look_thru = ["estaillades"]
reaction_phase = ["finalpressuredrop"]#,"initial_pressuredrop"]

# rootdir =r"H:\FlowHet_RxnDist\Menke2017\ket0.1ph3.6"#\GeoDict_Simulations"
# folders_to_look_thru = ["Menke2015","Menke2017","AlKhulaifi2018","AlKhulaifi2019"]#,"Hinz2019","PereiraNunes2016"]
# folders_to_look_thru = ["Pe_50","Pe_0.0005"]
# reaction_phase = ["Batch100"] #batchnumber

# rootdir = r"C:\Users\zkana\Downloads\data"
# folders_to_look_thru = ["60min","simi"]
# reaction_phase = []

if calc_components_single:
    for (root,dirs,files) in os.walk(rootdir):
        if any(folder in root for folder in folders_to_look_thru):
            imagesize = (498,498,324)
            for file in files:
                if ".mat" in file:
                    #file is the matlab file of the velocity components, need to make it magnitude
                    vel_components_file = root+"/"+file
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    Ux = []
                    Uy = []
                    Uz = []
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=False,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')

if pull_out_components:
    for root, dirs, files in os.walk(rootdir, topdown=False): 
        if "GeoDict_Simulations" not in root: continue
        if any(folder in root for folder in folders_to_look_thru) and any(phase in root for phase in reaction_phase):
            for file in files:
                if ".vap" in file:
                    guf_file = guf.GUF(os.path.join(root, file))
                    guf_image = guf_file.getImage("Velocity")
                    h5f = h5py.File(root + '/' + 'velocity_components.h5','w')
                    h5f.create_dataset('VelocityX', data = guf_image['VelocityX'], dtype ="float32")
                    h5f.create_dataset('VelocityY', data = guf_image['VelocityY'], dtype ="float32")
                    h5f.create_dataset('VelocityZ',data = guf_image['VelocityZ'], dtype ="float32")
                    h5f.close()
if calc_components:
    for (root,dirs,files) in os.walk(rootdir):
        if "GeoDict_Simulations" not in root: continue
        if any(folder in root for folder in folders_to_look_thru) and any(phase in root for phase in reaction_phase):
            sample = [x for x in df.index if x in root]
            imagesize = (int(df['nx'][sample].values[0]),int(df['ny'][sample].values[0]),int(df['nz'][sample].values[0]))
            phase = [phase for phase in reaction_phase if phase in root]
            print("converting to vel mag for ", sample, phase)
            for file in files:
                if ".h5" in file:
                    vel_components_file = root+"/"+file
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,loadfrommat=False,loadfromh5=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')
                elif "flowfield.mat" in file: 
                    #file is the matlab file of the velocity components, need to make it magnitude
                    vel_components_file = root+"/"+file
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')
                elif "Ux" in file and ".raw" in file:
                    vel_components_file = []
                    Ux = root + "/" + "Ux.raw"
                    Uy = root + "/" + "Uy.raw"
                    Uz = root +"/" + "Uz.raw"
                    vel_magnitude_file = root+"/vel_magnitude.raw"
                    if all([sample==["beadpack"],phase == ["final"]]): datatyp = 'float32'
                    else: datatyp = 'float64'
                    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,loadfrommat=False,loadfromraw=True,loadfromdat=False,datatype = datatyp)
                elif "Ux" in file and ".dat" in file: continue
                    # vel_components_file = []
                    # Ux = root + "/" + "Ux.dat"
                    # Uy = root + "/" + "Uy.dat"
                    # Uz = root +"/" + "Uz.dat"
                    # vel_magnitude_file = root+"/vel_magnitude.raw"
                    # vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=False,loadfrommat=False,loadfromraw=False,loadfromdat=True,datatype = 'float64')

for (root,dirs,files) in os.walk(rootdir):
    if "GeoDict_Simulations" not in root: continue
    if any(folder in root for folder in folders_to_look_thru) and any(phase in root for phase in reaction_phase):
        sample = [x for x in df.index if x in root]
        imagesize = (int(df['nx'][sample].values[0]),int(df['ny'][sample].values[0]),int(df['nz'][sample].values[0]))
        phase = [phase for phase in reaction_phase if phase in root]
        for file in files:
            if "vel_magnitude.raw" in file:
                print("calculating flow heterogeneity metrics ", sample,phase, root)
                #file is the matlab file of the velocity components, need to make it magnitude
                vel_magnitude_file = root+"/"+file
                structure_file = root+"/"+phase[0]+"_structure.raw"
                if calc_EMD:
                    distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=True,normalize_velocity_field=False,load_structure = False,plot = False,datatype = 'float32',logspacing=False,gen_ran_pop = False, compare_to_log=False,compare_to_Gauss=True)
                    print("manual-Gaussian EMD: ",distance)

                    distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=True,normalize_velocity_field=False,load_structure = False,plot = False,datatype = 'float32',logspacing=False,gen_ran_pop = False,compare_to_log=True,compare_to_Gauss=False)
                    print("manual-LogNormal EMD: ",distance)

                else: distance = []
                velocity_regions_file = root +"/_vel_regions.raw"
                if calc_percolation_threshold:
                    pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-4])
                    print(" pc: ",pc)
                else: pc = []
                if calc_V_S:
                    volume, surfacearea, ssa = dimensionless_volume_interfacialsurfacearea_SSA(velocity_regions_file,imagesize,include_disconnected_hvr=True,datatype = 'uint8')
                    print(" V_hv: ",volume, " S_hv: ", surfacearea, " SSA_hv: ",ssa)
                    volume, surfacearea, ssa = dimensionless_volume_interfacialsurfacearea_SSA(velocity_regions_file,imagesize,include_disconnected_hvr=False,datatype = 'uint8')
                    print(" V_perc: ",volume, " S_perc: ", surfacearea, " SSA_perc: ",ssa)
                else: 
                    volume = [] 
                    surfacearea = [] 
                    ssa = []
            # if "vel_magnitude_x.raw" in file:
            #     vel_magnitude_file = root+"/"+file
            #     velocity_regions_file = root +"/_vel_regions_x.raw"
            #     if calc_percolation_threshold:
            #         print("finding percolation threshold for ", sample,phase)
            #         pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-2])
            #         print(sample,phase,"in x pc: ",pc)
            #     else: pc = []
                # print(sample,phase," EMD: ",distance, " pc: ",pc, " V: ",volume, " S: ", surfacearea, " SSA: ",ssa)
# res.to_csv("EMD_variations_comparison.csv")
