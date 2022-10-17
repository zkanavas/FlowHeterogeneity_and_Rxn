import gd
from gd import stringmap
import pandas as pd
import os
import numpy as np
from csv import writer

def append_value(dict_obj, key, value):
    # Check if key exist in dict or not
    if key in dict_obj:
        # Key exist in dict.
        # Check if type of value of key is list or not
        if not isinstance(dict_obj[key], list):
            # If type is not list then make it list
            dict_obj[key] = [dict_obj[key]]
        # Append the value in list
        dict_obj[key].append(value)
    else:
        # As key is not in dict,
        # so, add key-value pair
        dict_obj[key] = value
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

create_sample_SSA_porosity_intime = True

directory = r'H:\FlowHet_RxnDist'
# allsampleinfo = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
intermstructureinfo = pd.read_csv("Publication_intermediate_structures.csv", header=0,index_col=1)

if create_sample_SSA_porosity_intime: 
    sample_SSA_porosity_intime = dict() 
else:
    sample_SSA_porosity_intime = pd.read_csv("sample_SSA_porosity_intime.csv",index_col = ["SampleName","SSA","Porosity"])

for root, dirs, files in os.walk(directory):
    if "GeoDict_Simulations" not in root: continue
    if "estaillades" not in root: continue
    if "Pe0.1" not in root: continue
    # elif "Hinz2019" in root: continue
    # elif "port0.1ph3.1" not in root: continue
    if "structures" in root:
        dirs[:] = []
        sample = [samp for samp in intermstructureinfo.index if samp in root][0]
        imagesize = (intermstructureinfo.nx[sample],intermstructureinfo.ny[sample],intermstructureinfo.nz[sample])
        voxelsize = intermstructureinfo.VoxelSize[sample]
        maxstep = intermstructureinfo.Steps[sample]
        
        for value in intermstructureinfo.Publication:
            if np.logical_and(value in root,value not in ["Menke2017","PereiraNunes2016"]):
                sample = intermstructureinfo[intermstructureinfo.Publication == value].index[0]
                nx = intermstructureinfo[intermstructureinfo.Publication == value].nx[sample]
                ny = intermstructureinfo[intermstructureinfo.Publication == value].ny[sample]
                nz = intermstructureinfo[intermstructureinfo.Publication == value].nz[sample]
                maxstep = intermstructureinfo[intermstructureinfo.Publication == value].Steps[sample]
                voxellength = intermstructureinfo[intermstructureinfo.Publication == value].VoxelSize[sample]
        if sample not in sample_SSA_porosity_intime:
            sample_SSA_porosity_intime[sample] = dict()
        
        #grab the .raw files, compute reff using current step structure relative to former step
        for step in range(0,maxstep):
            porosityfilename = root + "/Porosity"+str(step)+".gdr"
            surfacefilename = root + "/SurfaceArea"+str(step)+".gdr"
            originalGDR     = stringmap.parseGDR(porosityfilename)
            gdrDict         = originalGDR.toRecDict()
            porosity        = float(gdrDict['ResultMap']['OpenPorosity'])/100

            originalGDR     = stringmap.parseGDR(surfacefilename)
            gdrDict         = originalGDR.toRecDict()
            SSA        = float(gdrDict['ResultMap']['CountVoxelSurfaces']['SpecificSurfaceArea']) #m^2/m^3
            # print(sample,step,SSA,porosity)
            append_value(sample_SSA_porosity_intime[sample],"SSA",SSA)
            append_value(sample_SSA_porosity_intime[sample],"porosity",porosity)
        timestep = np.arange(36/60,((maxstep+1)*36)/60,36/60)
        if len(sample_SSA_porosity_intime[sample]["SSA"]) != len(timestep):
            chop = len(sample_SSA_porosity_intime[sample]["SSA"]) - len(timestep)
            timestep = timestep[:chop]
        assert len(sample_SSA_porosity_intime[sample]["SSA"]) == len(timestep)
        rowcontents = sample, intermstructureinfo.Publication[sample],voxellength, str(sample_SSA_porosity_intime[sample]["SSA"]),str(sample_SSA_porosity_intime[sample]["porosity"]),maxstep,list(timestep)
        append_list_as_row('sample_SSA_porosity_intime.csv', rowcontents)
# pd.DataFrame(sample_SSA_porosity_intime).to_csv('sample_SSA_porosity_intime.csv', index=False)