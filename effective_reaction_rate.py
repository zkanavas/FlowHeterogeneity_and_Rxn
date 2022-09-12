import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import csv

calc_rateofchange = False
calc_reff = True
plot_ = False

def singlemineral_reff(rho,M_mineral,S,phi_change,time_change):
    #assuming no grain porosity
    reff = ((rho)/(M_mineral*S))*(phi_change/time_change)
    return reff

def twominerals_reff(f_m1,f_m2,rho_m1,rho_m2,M_m1,M_m2,S,phi_change,time_change):
    #assuming no grain porosity
    reff = (phi_change/(S*time_change))*(((f_m1*rho_m1)/M_m1)+((f_m2*rho_m2)/M_m2))
    return reff

def check_convergence(roc, threshold=.0002):
    for ind,r in enumerate(roc[:-1]):
        if all(roc[ind:] < threshold):
            # print("converged")
            return ind
        else:
            continue

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

allsampleinfo = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
# intermstructureinfo = pd.read_csv("Publication_intermediate_structures.csv", header=0,index_col=1)
SSAporosityinfo = pd.read_csv("sample_SSA_porosity_intime.csv",header=0,index_col=0) #time is in minutes!!


#batch reaction rates, mol/m2s
rb_calcite = 8.1e-4 
rb_dolomite = 5.1e-5

#mineral densities, kg/m3
rho_calcite = 2.71e3 
rho_dolomite = 2.82e3

#molecular mass, kg/mol
M_calcite = 0.1001 
M_dolomite = 0.1844

if calc_reff:
    for sample in SSAporosityinfo.index:
        # if sample != "Sil_HetA_High":continue
        SSA = [float(ele) for ele in SSAporosityinfo.SSA[sample][1:-1].split(",")]
        porosity = [float(ele) for ele in SSAporosityinfo.Porosity[sample][1:-1].split(",")]
        timesteps = [int(ele) for ele in SSAporosityinfo.Timestep[sample][1:-1].split(",")]
        maxsteps = SSAporosityinfo.Steps[sample]
        reff = []
        print(sample)
        for step in range(1,maxsteps,1):
            #get specific surface area, porosity change, and time change from sample_SSA_porosity_intime.csv
            S = SSA[step-1]
            phi_change = porosity[step] - porosity[step-1]
            time_change = (timesteps[step]-timesteps[step-1])*60 #in seconds
            if "AlKhulaifi" not in SSAporosityinfo.Publication[sample]:
                #sample is considered to be pure calcite, use singlemineral_reff
                reff.append(singlemineral_reff(rho_calcite,M_calcite,S,phi_change,time_change))
            elif "AlKhulaifi2018" in SSAporosityinfo.Publication[sample]:
                #sample is considered to be pure calcite, use singlemineral_reff
                reff.append(singlemineral_reff(rho_dolomite,M_dolomite,S,phi_change,time_change))
            elif "AlKhulaifi2019" in SSAporosityinfo.Publication[sample]: 
                #sample is composite of calcite and dolormite, use twominerals_reff
                #get fraction (relative to all solids) of mineral
                #structure codes: pore space == 1, dolomite == 2, calcite == 3, filled edges ==4
                structurefile = r'H:/FlowHet_RxnDist/'+ SSAporosityinfo.Publication[sample] + "/" + sample + '/structures/'+ str(step)+".raw"
                structure = np.fromfile(structurefile, dtype=np.dtype('uint8')) 
                num_calcite = len(structure[structure==3])
                num_dolomite = len(structure[structure==2])
                f_calcite = num_calcite/(num_calcite+num_dolomite)
                f_dolomite = num_dolomite/(num_calcite+num_dolomite)  
                reff.append(twominerals_reff(f_calcite,f_dolomite,rho_calcite,rho_dolomite,M_calcite,M_dolomite,S,phi_change,time_change))
            else: print(SSAporosityinfo.Publication[sample])
        print(reff)

def moving_average(a, n=3) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# data = pd.read_csv(r"C:\Users\zkana\Downloads\menke2017.csv",header=[0,1])
# sampnames = ['est3.6','est3.1','ket3.6','ket3.1','port3.6','port3.1']
data = pd.read_csv(r"C:\Users\zkana\Downloads\alkhulaifi2018.csv",header=[0,1])
sampnames = ['HetBHigh','HetAHigh','HetALow','HetBLow']
fig,ax = plt.subplots()
averagewindow = 3
for sampname in sampnames:
    # if "ket" not in sampname: continue
    averagedreff = moving_average(data[sampname].Y.dropna())
    changeiny = abs(np.diff(averagedreff))/averagedreff[0]
    changeinx = abs(np.diff(data[sampname].X.dropna()[1:-1]))
    rateofchange = changeiny/changeinx
    [print(sampname,data[sampname].X.dropna()[ind+1]) for ind,value in enumerate(rateofchange) if value <3e-3]
    ax.plot(data[sampname].X.dropna(),data[sampname].Y,label=sampname)
    # print(rateofchange)


#find threshold
# xdata = [22.41495404,29.44332212,40.95373061,50.16001444,59.36940718,68.92260246,78.36003461,87.7974297,97.23502654,107.1006261,114.910408,123.2259297,131.6672342,141.3794975,152.3033352,164.1235812,176.6965228,188.864084,201.2749847,213.9367468]
# ydata = [3.235E-05,2.782E-05,2.11001E-05,1.55084E-05,1.02451E-05,1.44989E-05,1.99223E-05,2.53418E-05,3.07826E-05,2.78438E-05,2.21175E-05,1.62206E-05,1.02122E-05,4.47522E-06,6.13289E-06,7.98703E-06,1.03707E-05,9.42372E-06,7.38505E-06,5.04899E-06]
# diff = abs(np.diff(ydata))
# timestep = np.diff(xdata)
# rateofchange = diff/timestep
# print(rateofchange[16],rateofchange[17])
# for ind,val in enumerate(ydata):




# fig,ax = plt.subplots()
if calc_rateofchange: 
    for sample in SSAporosityinfo.index:    
        if "AlKhulaifi2018" not in SSAporosityinfo.Publication[sample]:continue
        # if "ket" not in sample: continue
        reff = [float(ele) for ele in SSAporosityinfo.reff[sample][1:-1].split(",")]
        timesteps = [int(ele) for ele in SSAporosityinfo.Timestep[sample][1:-1].split(",")] #in seconds
        averagedreff = moving_average(reff)
        changeiny = abs(np.diff(averagedreff))/averagedreff[0]
        changeinx = abs(np.diff(timesteps[2:-1]))
        rateofchange = changeiny/changeinx
        # [print(sample,timesteps[ind+2]) for ind,value in enumerate(rateofchange) if value <3e-3]
        ax.plot(timesteps[2:-1],averagedreff,label=sample)
    # timestep = np.diff(timesteps[1::])
    # diff = abs(np.diff(reff))
    # rateof_change = diff/timestep
    # percentchange = rateof_change/reff[0]
    # ind = check_convergence(percentchange)
    # if ind == None: 
    #     print(sample)
    # else: print(ind)
    # timesteps = [ele/60 for ele in timesteps]
    # ax.plot(timesteps[1::],reff,'-.', label=sample)
    
    # print(ind)
    # print("convergence at step ", ind, " with eff rxn rate ", reff[ind], "mol/m2s")
#         print(ind)
#         print("final rate of change:", (rateof_change)[-1]*100, "%")
    ax.set_ylim(0,1.3e-5)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Effective Reaction Rate (reff) [mol/m2s]")
    ax.legend()
    fig.tight_layout()
    plt.show()

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
# datafile = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\porosity_SSA_changes_allsimis.csv')
# changes = pd.read_csv(datafile,header=0)
# samplenames = changes.sample_name.unique()
# timestep = 3600/100 #s

# for samplename in samplenames:
#     if samplename == "estaillades":
#         phi_grain = 0
#     elif samplename == "ket0.1ph3.6":
#         phi_grain = (0.23-0.131)/(1-.131)
#     elif samplename == "ket0.1ph3.1":
#         phi_grain = (0.23-0.114)/(1-.114)
#     else: 
#         print("wrong sample name :", samplename)
#         continue
#     print("results for sample: ",samplename)
#     changeforsample = changes[changes.sample_name == samplename]
#     Pe = changeforsample.initial_Pe.unique()
    
#     for pe in Pe:
#         reff = []
#         print("Pe: ",pe)
#         changeforsamplePe = changeforsample[changeforsample.initial_Pe == pe]
#         changeforsamplePe.reset_index(inplace=True)
#         for ind in range(1,len(changeforsamplePe)):
#             phi_change = (changeforsamplePe.porosity[ind] - changeforsamplePe.porosity[ind-1])/100
#             reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changeforsamplePe.specific_surface_area[ind-1],phi_change,timestep))
#         print("average effective reaction rate: ", np.mean(reff))
#         diff = abs(np.diff(reff))
#         rateof_change = diff/reff[:-1]
#         ind = check_convergence(rateof_change)
#         print(ind)
#         print("final rate of change:", (rateof_change)[-1]*100, "%")

# # fig,ax = plt.subplots()
# # for ind in range(1,(len(changes))):
# #     phi_change = (changes.porosity[ind] - changes.porosity[ind-1])/100
# #     reff.append(r_eff(rho_calcite,phi_grain,M_calcite,changes.specific_surface_area[ind-1],phi_change,timestep))
# # if plot_:
# #     ax.scatter(changes.time_step[:-1],reff,c='k',marker='o')
# #     # ax.semilogy()
# #     ax.tick_params(axis='both',labelsize=14)
# #     ax.set_xlabel("Time, s", fontsize=15)
# #     ax.set_ylabel(r"$r_{eff}$, mol/m2s",fontsize=15)
# #     fig.tight_layout()





# if plot_:
#     plt.show()
    