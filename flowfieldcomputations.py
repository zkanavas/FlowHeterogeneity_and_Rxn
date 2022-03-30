import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mat73
from scipy import stats
import sklearn.metrics as metrics
from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import time

def convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize,Ux,Uy,Uz,clip_velocity_field=True,loadfrommat=True,loadfromraw=False,loadfromdat=False,datatype = 'float64'):
    #load images
    if loadfrommat: #use for .mat file type -- DONT NEED TO RESHAPE FROM .mat
        velocityfield = mat73.loadmat(vel_components_file)#directory + '/' + sample_descriptor + '.mat')
        #x-direction
        Ux_array = velocityfield['flowfield']['VelocityX']
        #y-direction
        Uy_array = velocityfield['flowfield']['VelocityY']
        # z-direction
        Uz_array = velocityfield['flowfield']['VelocityZ'] 
        if Ux_array.shape != imagesize: #remove first/last ten rows 
            Ux_array = Ux_array[:,:,9:imagesize[2]+9]
            Uy_array = Uy_array[:,:,9:imagesize[2]+9]
            Uz_array = Uz_array[:,:,9:imagesize[2]+9]
            
    elif loadfromraw:#use for .raw file type
        #x-direction
        Ux_array = np.fromfile(Ux, dtype=np.dtype('float32')) #specifying bc only this direction is in this datatype
        #y-direction
        Uy_array = np.fromfile(Uy, dtype=np.dtype(datatype))
        # z-direction
        Uz_array = np.fromfile(Uz, dtype=np.dtype(datatype)) 
        if clip_velocity_field: 
            #adjust image size
            imagesize[2]+=20
            #reshape
            Ux_array = Ux_array.reshape(imagesize) 
            Uy_array = Uy_array.reshape(imagesize) 
            Uz_array = Uz_array.reshape(imagesize)
            #reset image size
            imagesize[2]-=20
            #clip
            Ux_array = Ux_array[:,:,9:imagesize[2]+9]
            Uy_array = Uy_array[:,:,9:imagesize[2]+9]
            Uz_array = Uz_array[:,:,9:imagesize[2]+9]
        else: #just reshape
            Ux_array = Ux_array.reshape(imagesize) 
            Uy_array = Uy_array.reshape(imagesize) 
            Uz_array = Uz_array.reshape(imagesize)
    elif loadfromdat: #use for .dat file type
        #x-direction
        Ux_array = np.loadtxt(Ux) 
        #y-direction
        Uy_array = np.loadtxt(Uy) 
        #z-direction
        Uz_array = np.loadtxt(Uz) 
        if clip_velocity_field: 
            #adjust image size
            imagesize[2]+=20
            #reshape
            Ux_array = Ux_array.reshape(imagesize) 
            Uy_array = Uy_array.reshape(imagesize) 
            Uz_array = Uz_array.reshape(imagesize)
            #reset image size
            imagesize[2]-=20
            #clip
            Ux_array = Ux_array[:,:,9:imagesize[2]+9]
            Uy_array = Uy_array[:,:,9:imagesize[2]+9]
            Uz_array = Uz_array[:,:,9:imagesize[2]+9]
        else: #just reshape
            Ux_array = Ux_array.reshape(imagesize) 
            Uy_array = Uy_array.reshape(imagesize) 
            Uz_array = Uz_array.reshape(imagesize)
    
    #calculate magnitude
    vel_magnitude = sum([Ux_array**2, Uy_array**2, Uz_array**2])**(1/2)

    #save magnitude file
    vel_magnitude.astype('float32').tofile(vel_magnitude_file)#directory +"/" + sample_descriptor + '_velocity_magnitude.raw')
    return vel_magnitude

def earth_movers_distance(vel_magnitude_file,imagesize,structure_file,manually_compute=False,normalize_velocity_field=False,load_structure = True,plot = False,datatype = 'float32',logspacing=True):
    rng = np.random.default_rng(203)
    #load velocity magnitude
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    #remove structure
    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(imagesize)
        # structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape((imagesize[0],imagesize[1],imagesize[2]))
        #remove grains
        vel_magnitude = vel_magnitude[structure == 0]
    else:
        vel_magnitude = vel_magnitude[vel_magnitude != 0]
    
    #normalizing velocity field by mean
    if normalize_velocity_field:
        mean = np.mean(vel_magnitude)
        vel_magnitude /= mean
    # extract sample statistics
    mean = np.mean(vel_magnitude)
    std = np.std(vel_magnitude)
    
    #generate homogeneous (log-normal) distribution
    generated = rng.lognormal(mean=mean,sigma = std, size=1000)#vel_magnitude.size)
    
    if manually_compute:
        bin_min = np.min([np.min(vel_magnitude),np.min(generated)])/2
        bin_max = np.max([np.max(vel_magnitude),np.max(generated)])
        if logspacing:
            bins = 10 ** np.linspace(np.log10(bin_min), np.log10(bin_max),num=1000)
        else:
            bins = np.linspace(bin_min,bin_max,num=1000)
        pdf,velmag_bins = np.histogram(vel_magnitude,density=True,bins = bins) 
        cumulsum = np.cumsum(pdf)
        velmag_cdf = cumulsum/cumulsum[-1]
        pdf,gen_bins = np.histogram(generated,density=True,bins = bins) 
        cumulsum = np.cumsum(pdf)
        gen_cdf = cumulsum/cumulsum[-1]

        # if gen_bins[-2] < velmag_bins[-2]:
        #     gen_bins = np.append(gen_bins,[velmag_bins[-2],velmag_bins[-1]]) 
        #     gen_cdf = np.append(gen_cdf,[1,1])
        # else:
        #     velmag_bins = np.append(velmag_bins,[gen_bins[-2],gen_bins[-1]])
        #     velmag_cdf = np.append(velmag_cdf,[1,1])

        # if gen_bins[0] < velmag_bins[0]:
        #     velmag_cdf = np.insert(velmag_cdf,0,0)
        #     velmag_bins = np.insert(velmag_bins, 0,np.min(gen_bins)) 
        # else:
        #     gen_cdf = np.insert(gen_cdf,0,0)
        #     gen_bins = np.insert(gen_bins, 0,np.min(velmag_bins)) 
        distance = abs(metrics.auc(velmag_bins[:-1],velmag_cdf) - metrics.auc(gen_bins[:-1],gen_cdf))
    else:
        distance = stats.wasserstein_distance(vel_magnitude,generated)
    if plot == True:
        fig,ax = plt.subplots()
        ax.plot(velmag_bins[:-1],velmag_cdf,label='true distribution')
        ax.plot(gen_bins[:-1],gen_cdf,label='log-normal distribution')
        ax.semilogx()
        ax.tick_params(axis='both',labelsize=14)
        ax.set_xlabel('V/<V>', fontsize=15)
        ax.set_ylabel('CDF',fontsize=15)
        ax.legend()
        fig.tight_layout()
    if plot == True: plt.show()
    return distance

def checkbounds(vel_normalized,percolation_threshold,imagesize):
    vel_norm = np.zeros(vel_normalized.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','area'])
    props = pd.DataFrame(props)
    checking_bounds = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    return any(checking_bounds)

def save(vel_normalized,percolation_threshold,imagesize,velocity_regions_file):
    vel_norm = np.zeros(vel_normalized.shape)
    [x_2,y_2,z_2] = np.where(vel_normalized >= percolation_threshold)
    vel_norm[x_2,y_2,z_2] = 1
    labels_out = label(vel_norm)
    props = regionprops_table(labels_out,properties =['label','bbox','coords','area'])
    props = pd.DataFrame(props)
    id_box = props['bbox-5'][props['bbox-2']==0] == imagesize[2]
    coords = props['coords'][id_box.index[np.where(id_box)]].tolist()
    #id velocity zones
    vel_norm[x_2,y_2,z_2] = 2 #disconnected high velocity region
    vel_norm[coords[0][:,0],coords[0][:,1],coords[0][:,2]] = 3 #percolating path
    [x_1,y_1,z_1] = np.where(np.logical_and(vel_normalized < percolation_threshold, vel_normalized > 0))
    vel_norm[x_1,y_1,z_1] = 1 #stagnant zone
    #the leftovers are 0 and correspond to solid

    #save thresholded velocity field
    vel_norm.astype('uint8').tofile(velocity_regions_file)#directory +"/" + sample_descriptor + '_velocity_regions.txt')

def percolation_threshold(vel_magnitude_file,imagesize,velocity_regions_file,structure_file,load_structure=True,save_regions=True,datatype='float32',tolerance = [1e-2]):

    #load images
    vel_magnitude = np.fromfile(vel_magnitude_file, dtype=np.dtype(datatype)) 
    vel_magnitude = vel_magnitude.reshape(imagesize)

    if load_structure == True:
        vel_magnitude = vel_magnitude.reshape(imagesize)
        # structure_file = directory + "/" + sample + "_structure.raw"
        structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
        structure = structure.reshape((imagesize[0],imagesize[1],imagesize[2]))
        #remove grains
        vel_magnitude_nostructure = vel_magnitude[structure != 1]
    else:
        vel_magnitude_nostructure = vel_magnitude[vel_magnitude != 0]
    mean = np.mean(vel_magnitude_nostructure)
    vel_normalized = np.divide(vel_magnitude,mean)  
    percolation_threshold = np.max(vel_normalized)

    upper_pt = percolation_threshold
    lower_pt = upper_pt/2
    stop = False
    while not stop:
        # print(upper_pt,lower_pt)
        if not checkbounds(vel_normalized,lower_pt,imagesize):
            upper_pt_new = lower_pt
            lower_pt -= (upper_pt-lower_pt)/2 
            upper_pt = upper_pt_new
        else:
            difference = upper_pt - lower_pt
            if difference < tolerance:
                # print(sample_descriptor,' final pc: ',lower_pt)
                if save_regions: save(vel_normalized,lower_pt,imagesize,velocity_regions_file)
                stop=True
            else: 
                lower_pt += (upper_pt-lower_pt)/2
    return lower_pt

def dimensionless_volume_interfacialsurfacearea_SSA(velocity_regions_file,imagesize,include_disconnected_hvr=True,datatype = 'uint8'):
  
    #load image
    npimg = np.fromfile(velocity_regions_file, dtype=np.dtype(datatype)) 
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

    #load image
    npimg = np.fromfile(velocity_regions_file, dtype=np.dtype(datatype)) 
    npimg = npimg.reshape(imagesize)
    if include_disconnected_hvr: npimg[npimg == 2] = 3 #includes disconnected high velocity region
    npimg[npimg != 3] = 0 
    labels_out = label(npimg)

    #convert to mesh format
    verts, faces, normals, values = marching_cubes(labels_out,level=0)

    #find surface area
    surfacearea_perc = mesh_surface_area(verts, faces)

    #find total volume
    volume_perc = len(npimg[npimg!=0])

    surfacearea = (surfacearea_perc/surfacearea_grains)
    volume = (volume_perc/void_volume)
    specificsurfacearea=surfacearea/volume

    return volume,surfacearea,specificsurfacearea