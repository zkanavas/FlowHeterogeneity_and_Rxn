import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

plot_ = False

def r_eff(rho,phi_grain,M_mineral,S,phi_change,time_change):
    reff = ((rho*(1-phi_grain))/(M_mineral*S))*(phi_change/time_change)
    return reff


# structure_file = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\structures\Porositytime0\sample_sleeve_comb.raw')

# struct = np.fromfile(structure_file, dtype=np.dtype('int8'))
# struct = struct.reshape((498,498,324)) 

porosity_time = [(5.96,0),(7.20,12),(7.74,37),(9.44,62)]
porosity_time = [(5.96,0),(9.44,62)]
S = [5649.93,6327.19,6107.66]
# pubS = [7410,8940,9170]
time_in_seconds = False
time_convert = 60 #60s in 1 min
phi_grain = (0.234-0.0938)/(1-0.0938)
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

datafile = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\baseline_100batches_v2\porosity_SSA_changes.csv')
changes = pd.read_csv(datafile,header=0)
timestep = 3600/100 #s
reff = []
fig,ax = plt.subplots()
for ind in range(1,(len(changes))):
    phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
    reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,timestep))
if plot_:
    ax.scatter(changes.time_step[:-1],reff,c='k',marker='o')
    # ax.semilogy()
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel("Time, s", fontsize=15)
    ax.set_ylabel(r"$r_{eff}$, mol/m2s",fontsize=15)
    fig.tight_layout()
diff = abs(np.diff(reff))
rateof_change = diff/reff[-1]

print("final rate of change: ", abs(np.diff(rateof_change))[-1]*100, "%")

print('next')

datafile = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2015\structures\porosity_SSA_changes.csv')
changes = pd.read_csv(datafile,header=0)
phi_grain = (0.27-(changes.porosity[0]/100))/(1-(changes.porosity[0]/100))
time_convert=60
reff=[]
fig,ax = plt.subplots()
for ind in range(1,(len(changes))):
    phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
    time_change = (changes.time_step[ind] - changes.time_step[ind-1])*time_convert #now in seconds
    reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,time_change))
if plot_:
    ax.scatter(changes.time_step[:-1],reff,c='k',marker='o')
    # ax.semilogy()
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel("Time, s", fontsize=15)
    ax.set_ylabel(r"$r_{eff}$, mol/m2s",fontsize=15)
    fig.tight_layout()

diff = abs(np.diff(reff))
rateof_change = diff/reff[-1]
print("final rate of change: ", abs(np.diff(rateof_change))[-1]*100, "%")

print('next')

datafile = os.path.normpath(r'D:\FlowHet_RxnDist\Menke2017\ket0.1ph3.1\structures\porosity_SSA_changes.csv')
changes = pd.read_csv(datafile,header=0)
phi_grain = (0.234-(changes.porosity[0]/100))/(1-(changes.porosity[0]/100))
time_convert=60
reff=[]
fig,ax = plt.subplots()
for ind in range(1,(len(changes))):
    phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
    time_change = (changes.time_step[ind] - changes.time_step[ind-1])*time_convert #now in seconds
    reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,time_change))
if plot_:
    ax.scatter(changes.time_step[:-1],reff,c='k',marker='o')
    # ax.semilogy()
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel("Time, s", fontsize=15)
    ax.set_ylabel(r"$r_{eff}$, mol/m2s",fontsize=15)
    fig.tight_layout()

diff = abs(np.diff(reff))
rateof_change = diff/reff[-1]
print("final rate of change: ", abs(np.diff(rateof_change))[-1]*100, "%")

if plot_:
    plt.show()
    