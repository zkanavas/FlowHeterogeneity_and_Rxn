from flowfieldcomputations import convert_components_into_magnitude,earth_movers_distance,percolation_threshold,dimensionless_volume_interfacialsurfacearea_SSA

# required libraries: numpy, pandas, matplotlib, mat73, scipy, sklearn, skimage

vel_components_file = 
vel_magnitude_file = 
imagesize = 
velocity_regions_file = 
structure_file =
Ux = 
Uy = 
Uz = 

convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=True,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64')

distance = earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=False,normalize_velocity_field=False,load_structure = False,plot = False,datatype = 'float32')

pc = percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=False,save_regions=True,datatype='float32',tolerance = [1e-2])

volume, surfacearea, ssa = dimensionless_volume_interfacialsurfacearea_SSA(velocity_regions_file,imagesize,include_disconnected_hvr=True,datatype = 'uint8')

