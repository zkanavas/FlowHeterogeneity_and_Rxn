#To load virtual environment: .venv\Scripts\activate.bat
import flowfieldcomputations 
import os
import pandas as pd
import gd
from gd import stringmap
from gd import guf
import h5py

calc_components = False #this would be the matlab step
calc_vel_mag = False
calc_pc = False
sample_information = pd.read_csv('sample_information.csv', header = 0, index_col= 0)
sample_information_pc = pd.read_csv('sample_information_pc.csv', header = 0, index_col= 0)
PCs = [7.972, 7.6404, 6.6242, 4.7524, 5.6903, 4.3965, 3.8688]
if "pc" not in sample_information.columns:
    sample_information.insert(len(sample_information.columns),'pc',[[],[],[],[],[],[],[]])
if calc_components:
    for root, dirs, files in os.walk("D:\JW\evolving_structures\port0.1ph3.1", topdown=False): #Change this back to "." instead of full location
        for file in files:
            if ".vap" in file:
                guf_file = guf.GUF(os.path.join(root, file))
                guf_image = guf_file.getImage("Velocity")
                h5f = h5py.File(root + '/' + 'velocity_components.h5','w')
                h5f.create_dataset('VelocityX', data = guf_image['VelocityX'], dtype ="float32")
                h5f.create_dataset('VelocityY', data = guf_image['VelocityY'], dtype ="float32")
                h5f.create_dataset('VelocityZ',data = guf_image['VelocityZ'], dtype ="float32")
                h5f.close()
for root, dirs, files in os.walk("D:\JW\evolving_structures\port0.1ph3.1", topdown=False): #Change this back to "." instead of full location
    for file in files:
        if ".h5" in file:
            vel_components_file = os.path.join(root, file) #equivalent to vel_components_file = root + "/" + file
            vel_magnitude_file = os.path.join(root,"vel_mag.raw")
            vel_region = os.path.join(root,"vel_reg.raw")
            samplename = [x for x in sample_information.index if x in root][0]
            #for samplename in sample_information.index:
                #if samplename in root:
            imagesize = (sample_information.nx[samplename], sample_information.ny[samplename],sample_information.nz[samplename])
            print(samplename)
            if calc_vel_mag:
                print("Vel mag start")
                flowfieldcomputations.convert_components_into_magnitude(vel_components_file,vel_magnitude_file,imagesize)
            if calc_pc:
                print("pc start")
                PC= flowfieldcomputations.percolation_threshold(vel_magnitude_file,imagesize,vel_region)
                sample_information_pc.pc[samplename].append(PC)
                print(round(PC,4))
samplename = 'port0.1ph3.1'
sample_information_pc.pc[samplename] = []
PCs = [7.972, 7.6404, 6.6242, 4.7524, 5.6903, 4.3965, 3.8688]
for i in PCs:
    sample_information_pc.pc[samplename].append(i)

#sample_information_pc.loc[samplename] = {'pc':PC}
#sample_information_pc.append(sample_information[samplename])
sample_information_pc.to_csv('sample_information_pc.csv')
       