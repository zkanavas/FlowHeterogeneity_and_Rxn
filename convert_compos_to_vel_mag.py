import os
import numpy as np

#customize these for each sample
sample_descriptor = "estaillades"
imagesize =(650,650,650)
datatype = 'float64'

#data directory
directory = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\Data')

#data files
Ux_velfield = directory + '/Ux_estaillades.dat'
Uy_velfield = directory + '/Uy_estaillades.dat'
Uz_velfield = directory + '/Uz_estaillades.dat'

#load images
#x-direction
# Ux_array = np.fromfile(Ux_velfield, dtype=np.dtype(datatype)) #use for .raw file type
Ux_array = np.loadtxt(Ux_velfield) #use for .dat file type
Ux_array = Ux_array.reshape(imagesize) #reshape

#y-direction
# Uy_array = np.fromfile(Uy_velfield, dtype=np.dtype(datatype)) #use for .raw file type
Uy_array = np.loadtxt(Uy_velfield) #use for .dat file type
Uy_array = Uy_array.reshape(imagesize) #reshape

#z-direction
# Uz_array = np.fromfile(Uz_velfield, dtype=np.dtype(datatype)) #use for .raw file type
Uz_array = np.loadtxt(Uz_velfield) #use for .dat file type
Uz_array = Uz_array.reshape(imagesize) #reshape

#calculate magnitude
vel_magnitude = sum([Ux_array**2, Uy_array**2, Uz_array**2])**(1/2)

#save magnitude file
vel_magnitude.astype('float16').tofile(directory +"/" + sample_descriptor + '_velocity_magnitude.txt')
