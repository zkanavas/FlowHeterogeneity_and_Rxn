from flowfieldcomputations import convert_components_into_magnitude,earth_movers_distance,percolation_threshold,dimensionless_volume_interfacialsurfacearea_SSA
# required libraries: numpy, pandas, matplotlib, mat73, scipy, sklearn, skimage
import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

rootdir = r"D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.6"
folders_to_look_thru = ["Pe50"]
batch = "Batch100"
# folders_to_look_thru = ["Pe0.5_cont"]
# batch = "Batch75"
imagesize = (499,499,450)
for folder in folders_to_look_thru:
    #file is the matlab file of the velocity components, need to make it magnitude
    vel_components_file = rootdir+"/"+folder+"/"+batch+"/final_pressuredrop.mat"
    vel_magnitude_file = rootdir+"/"+folder+"/"+batch+"/vel_magnitude.raw"
    Ux = []
    Uy = []
    Uz = []
    vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=True,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')  
    
    structure_file = []
    velocity_regions_file = rootdir+"/"+folder+"/"+batch+"/_vel_regions.raw"
    print("finding percolation threshold for ", folder)
    pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-2])
    print(folder,pc)