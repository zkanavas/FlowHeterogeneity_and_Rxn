import os
import numpy as np
import mat73
import time
import matplotlib.pyplot as plt


#customize these for each sample
# sample_descriptor = "Sil_HetA_High_Scan1"
# imagesize =(839,849,812)

# sample_descriptor = "Sil_HetA_Low_Scan1"
# imagesize =(936,936,787)

# sample_descriptor = "Sil_HetB_High_Scan1"
# imagesize =(911,914,829)

sample_descriptor = "Sil_HetB_Low_Scan1"
imagesize =(903,889,785)

datatype = 'float64'

#data directory
directory = os.path.normpath(r'F:\FlowHet_RxnDist')

# #data files
# Ux_velfield = directory + '/Ux_estaillades.dat'
# Uy_velfield = directory + '/Uy_estaillades.dat'
# Uz_velfield = directory + '/Uz_estaillades.dat'
# f = h5py.File(directory + '/' + sample_descriptor + ".mat") #use for .mat file type
velocityfield = mat73.loadmat(directory + '/' + sample_descriptor + '.mat')
# velocityfield = np.fromfile(directory + '/' + sample_descriptor + '_Umag.raw',dtype=np.dtype(datatype))

# tic = time.perf_counter()

#load images
#x-direction
# Ux_array = np.fromfile(Ux_velfield, dtype=np.dtype(datatype)) #use for .raw file type
# Ux_array = np.loadtxt(Ux_velfield) #use for .dat file type
# Ux_array = Ux_array.reshape(imagesize) #reshape
Ux_array = velocityfield['flowfield']['VelocityX'] #use for .mat file type -- DONT NEED TO RESHAPE FROM .mat

#y-direction
# Uy_array = np.fromfile(Uy_velfield, dtype=np.dtype(datatype)) #use for .raw file type
# Uy_array = np.loadtxt(Uy_velfield) #use for .dat file type
# Uy_array = Uy_array.reshape(imagesize) #reshape
Uy_array = velocityfield['flowfield']['VelocityY'] #use for .mat file type -- DONT NEED TO RESHAPE FROM .mat

#z-direction
# Uz_array = np.fromfile(Uz_velfield, dtype=np.dtype(datatype)) #use for .raw file type
# Uz_array = np.loadtxt(Uz_velfield) #use for .dat file type
# Uz_array = Uz_array.reshape(imagesize) #reshape
Uz_array = velocityfield['flowfield']['VelocityZ'] #use for .mat file type -- DONT NEED TO RESHAPE FROM .mat

#calculate magnitude
vel_magnitude = sum([Ux_array**2, Uy_array**2, Uz_array**2])**(1/2)

#save magnitude file
vel_magnitude.astype('float32').tofile(directory +"/" + sample_descriptor + '_velocity_magnitude.raw')

# toc = time.perf_counter()
# print("time elapsed: " + str(toc-tic) + " seconds" )
