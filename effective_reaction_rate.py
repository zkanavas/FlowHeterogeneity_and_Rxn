from more_itertools import sample
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

plot_ = False

def r_eff(rho,phi_grain,M_mineral,S,phi_change,time_change):
    reff = ((rho*(1-phi_grain))/(M_mineral*S))*(phi_change/time_change)
    return reff

def check_convergence(roc, threshold=.2):
    for ind,r in enumerate(roc):
        if all(roc[ind:] < threshold):
            # print("converged")
            return ind
        else:
            continue
# structure_file = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\structures\Porositytime0\sample_sleeve_comb.raw')

# struct = np.fromfile(structure_file, dtype=np.dtype('int8'))
# struct = struct.reshape((498,498,324)) 

# porosity_time = [(5.96,0),(7.20,12),(7.74,37),(9.44,62)]
# porosity_time = [(5.96,0),(9.44,62)]
# S = [5649.93,6327.19,6107.66]
# pubS = [7410,8940,9170]
time_in_seconds = False
time_convert = 60 #60s in 1 min
phi_grain = 0#(0.234-0.0938)/(1-0.0938)
r = 8.1e-4 #batch reaction rate
rho_calcite = 2.71e3 #[kg m-3]
M_calcite = 0.1 #[kg mol-1]

# reff=[]
# for ind in range(1,len(porosity_time)):
#     phi_change = (porosity_time[ind][0] - porosity_time[ind-1][0])/100
#     time_change = (porosity_time[ind][1] - porosity_time[ind-1][1])*time_convert #now in seconds
#     reff.append(r_eff(rho_calcite,phi_grain,M_calcite,S[ind-1],phi_change,time_change))
#     print(reff)
# diff = np.diff(reff)
# percent_change = diff/reff[:-1]
# print("final percent change: ", percent_change[-1]*100, "%")

# datafile = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\baseline_100batches_v2\porosity_SSA_changes.csv')
datafile = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\porosity_SSA_changes_allsimis.csv')
changes = pd.read_csv(datafile,header=0)
samplenames = changes.sample_name.unique()
timestep = 3600/100 #s

for samplename in samplenames:
    if samplename == "estaillades":
        phi_grain = 0
    elif samplename == "ket0.1ph3.6":
        phi_grain = (0.23-0.131)/(1-.131)
    elif samplename == "ket0.1ph3.1":
        phi_grain = (0.23-0.114)/(1-.114)
    else: 
        print("wrong sample name :", samplename)
        continue
    print("results for sample: ",samplename)
    changeforsample = changes[changes.sample_name == samplename]
    Pe = changeforsample.initial_Pe.unique()
    
    for pe in Pe:
        reff = []
        print("Pe: ",pe)
        changeforsamplePe = changeforsample[changeforsample.initial_Pe == pe]
        changeforsamplePe.reset_index(inplace=True)
        for ind in range(1,len(changeforsamplePe)):
            phi_change = (changeforsamplePe.porosity[ind] - changeforsamplePe.porosity[ind-1])/100
            reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changeforsamplePe.specific_surface_area[ind-1],phi_change,timestep))
        print("average effective reaction rate: ", np.mean(reff))
        diff = abs(np.diff(reff))
        rateof_change = diff/reff[:-1]
        ind = check_convergence(rateof_change)
        print(ind)
        print("final rate of change:", (rateof_change)[-1]*100, "%")

# fig,ax = plt.subplots()
# for ind in range(1,(len(changes))):
#     phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
#     reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,timestep))
# if plot_:
#     ax.scatter(changes.time_step[:-1],reff,c='k',marker='o')
#     # ax.semilogy()
#     ax.tick_params(axis='both',labelsize=14)
#     ax.set_xlabel("Time, s", fontsize=15)
#     ax.set_ylabel(r"$r_{eff}$, mol/m2s",fontsize=15)
#     fig.tight_layout()





if plot_:
    plt.show()
    