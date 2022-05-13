import numpy as np
import os
import pandas as pd


def r_eff(rho,phi_grain,M_mineral,S,phi_change,time_change):
    reff = ((rho*(1-phi_grain))/(M_mineral*S))*(phi_change/time_change)
    return reff


# structure_file = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\structures\Porositytime0\sample_sleeve_comb.raw')

# struct = np.fromfile(structure_file, dtype=np.dtype('int8'))
# struct = struct.reshape((498,498,324)) 

# porosity_time = [(5.96,0),(7.20,12),(7.74,37),(9.44,62)]
# porosity_time = [(5.96,0),(9.44,62)]
# S = [5649.93,6327.19,6107.66]
# pubS = [7410,8940,9170]
time_in_seconds = False
time_convert = 60 #60s in 1 min
phi_grain = (0.234-0.0938)/(1-0.0938)
print(phi_grain)
r = 8.1e-4 #batch reaction rate
rho_calcite = 2.71e3 #[kg m-3]
M_calcite = 0.1 #[kg mol-1]

# print(range(1,len(porosity_time)))
# for ind in range(1,len(porosity_time)):
#     phi_change = (porosity_time[ind][0] - porosity_time[ind-1][0])/100
#     time_change = (porosity_time[ind][1] - porosity_time[ind-1][1])*time_convert #now in seconds
#     reff= r_eff(rho_calcite,phi_grain,M_calcite,S[ind-1],phi_change,time_change)
#     print(reff)

datafile = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\baseline_100batches_v2\porosity_SSA_changes.csv')

changes = pd.read_csv(datafile,header=0)
timestep = 3600/100 #s

for ind in range(1,(len(changes))):
    phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
    reff= r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,timestep)
    print(reff)

