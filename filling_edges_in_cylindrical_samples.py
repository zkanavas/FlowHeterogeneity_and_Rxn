import os
import numpy as np

rootdir =r"H:\FlowHet_RxnDist"

for (root,dirs,files) in os.walk(rootdir):
    if "AlKhulaifi" in root:
        for file in files:
            if ".raw" in file:
                structure_file = root+"/"+file
                structure = np.fromfile(structure_file,dtype=np.dtype('uint8'))
                structure[structure == 0] = 4 #changing corners to unique material ID
                structure.astype('uint8').tofile(root +"/" + file[:-4] + '_structure.raw')
                print(file[:-4],"done")
