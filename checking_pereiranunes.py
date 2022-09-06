from flowfieldcomputations import convert_components_into_magnitude,earth_movers_distance,percolation_threshold,dimensionless_volume_interfacialsurfacearea_SSA
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = r'H:\FlowHet_RxnDist\PereiraNunes2016\beadpack\initial'
sample = "beadpack"
phase = "initial"
imagesize = (500,500,500)
vel_components_file = []
Ux = root + "/" + "Ux.raw"
Uy = root + "/" + "Uy.raw"
Uz = root +"/" + "Uz.raw"
vel_magnitude_file = root+"/vel_magnitude.raw"
if all([sample==["beadpack"],phase == ["final"]]): datatyp = 'float32'
else: datatyp = 'float64'
# vel_magnitude=convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=False,loadfrommat=False,loadfromraw=True,loadfromdat=False,datatype = datatyp)

velocity_regions_file = root +"/vel_regions2.raw"
structure_file=[]
pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-2])
print(sample,phase,"pc: ",pc)