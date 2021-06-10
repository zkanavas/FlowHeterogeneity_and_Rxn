import os
import numpy as np

#customize these for each sample
sample_descriptor = "beadpack"
imagesize =(500,500,500)
datatype = 'float64'

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\PracticeFolder')

#data files
Ux_velfield = directory + '/Ux_beadpack_t0.raw'
Uy_velfield = directory + '/Uy_beadpack_t0.raw'
Uz_velfield = directory + '/Uz_beadpack_t0.raw'

#load images
Ux_array = np.fromfile(Ux_velfield, dtype=np.dtype(datatype)) #x-direction
Ux_array = Ux_array.reshape(imagesize)
Uy_array = np.fromfile(Uy_velfield, dtype=np.dtype(datatype)) #y-direction
Uy_array = Uy_array.reshape(imagesize)
Uz_array = np.fromfile(Uz_velfield, dtype=np.dtype(datatype)) #z-direction
Uz_array = Uz_array.reshape(imagesize)

#calculate magnitude
vel_magnitude = sum([Ux_array**2, Uy_array**2, Uz_array**2])**(1/2)

#save magnitude file
# vel_magnitude.astype('float16').tofile(sample_descriptor + '_velocity_magnitude.txt')

mean = np.mean(vel_magnitude[vel_magnitude != 0])
vel_norm = np.divide(vel_magnitude,mean)
percolation_threshold = np.max(vel_norm)


[x_2,y_2,z_2] = np.where(vel_norm >= percolation_threshold)
[x_1,y_1,z_1] = np.where(np.logical_and(vel_norm < percolation_threshold, vel_norm != 0))
vel_norm[x_2,y_2,z_2] = 2
vel_norm[x_1,y_1,z_1] = 1

#start to incrementally lower percolation_threshold until a continuous of region of high velocity forms
while vel_norm ==2 not continuous across x:
    lower percolation_threshold
    check for continuity
    
ddd