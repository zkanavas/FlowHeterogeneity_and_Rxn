import gd
from gd import stringmap
import pandas as pd
import os

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


create_sample_SSA_porosity_intime = True

directory = r'H:\FlowHet_RxnDist'
allsampleinfo = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
intermstructureinfo = pd.read_csv("Publication_intermediate_structures.csv", header=0,index_col=1)

if create_sample_SSA_porosity_intime: 
        sample_SSA_porosity_intime = dict() 
else:
    sample_SSA_porosity_intime = pd.read_csv("sample_SSA_porosity_intime.csv",index_col = ["SampleName","SSA","Porosity"])

for root, dirs, files in os.walk(directory):
    if "PereiraNunes2016" in root: continue
    elif "Hinz2019" in root: continue
    elif "port0.1ph3.1" not in root: continue
    if "structures" in root:
        dirs[:] = []
        sample = [samp for samp in intermstructureinfo.index if samp in root][0]
        if sample not in sample_SSA_porosity_intime:
            sample_SSA_porosity_intime[sample] = dict()
        imagesize = (allsampleinfo.nx[sample],allsampleinfo.ny[sample],allsampleinfo.nz[sample])
        voxelsize = intermstructureinfo.VoxelSize[sample]
        #grab the .raw files, compute reff using current step structure relative to former step
        num = intermstructureinfo.Steps[sample]
        for step in range(0,num):
            porosityfilename = root + "/Porosity"+str(step)+".gdr"
            surfacefilename = root + "/SurfaceArea"+str(step)+".gdr"
            originalGDR     = stringmap.parseGDR(porosityfilename)
            gdrDict         = originalGDR.toRecDict()
            porosity        = float(gdrDict['ResultMap']['OpenPorosity'])/100

            originalGDR     = stringmap.parseGDR(surfacefilename)
            gdrDict         = originalGDR.toRecDict()
            SSA        = float(gdrDict['ResultMap']['CountVoxelSurfaces']['SpecificSurfaceArea']) #m^2/m^3
            print(sample,step,SSA,porosity)
            # append_value(sample_SSA_porosity_intime[sample],"SSA",SSA)
            # append_value(sample_SSA_porosity_intime[sample],"porosity",porosity)

# pd.DataFrame(sample_SSA_porosity_intime).to_csv('sample_SSA_porosity_intime.csv', index=False)