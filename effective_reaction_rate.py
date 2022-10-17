import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table, marching_cubes, mesh_surface_area
import csv


calc_reff = False
savenewfile = True
calc_rateofchange = True
plot_ = False

def singlemineral_reff(rho,M_mineral,S,phi_change,time_change):
    #assuming no grain porosity
    reff = ((rho)/(M_mineral*S))*(phi_change/time_change)
    return reff

def twominerals_reff(f_m1,f_m2,rho_m1,rho_m2,M_m1,M_m2,S,phi_change,time_change):
    #assuming no grain porosity
    reff = (phi_change/(S*time_change))*(((f_m1*rho_m1)/M_m1)+((f_m2*rho_m2)/M_m2))
    return reff

def check_convergence(roc_array,threshold=3e-3):
    for ind,r in enumerate(roc_array):
        if all(roc_array[ind:] < threshold):
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

def moving_average(a, n=2) :
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# allsampleinfo = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
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
        if sample != "estaillades_sim2":continue
        # if SSAporosityinfo.Publication[sample] != "Pe_0.0005": continue
        SSA = [float(ele) for ele in SSAporosityinfo.SSA[sample][1:-1].split(",")]
        porosity = [float(ele) for ele in SSAporosityinfo.Porosity[sample][1:-1].split(",")]
        timesteps = [float(ele) for ele in SSAporosityinfo.Timestep[sample][1:-1].split(",")]
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
        SSAporosityinfo.reff[sample] = reff
        print(reff)
    if savenewfile:
        pd.DataFrame(SSAporosityinfo).to_csv('sample_SSA_porosity_intime.csv', index=True)

if plot_:
    fig,ax = plt.subplots()
if calc_rateofchange: 
    for sample in SSAporosityinfo.index:    
        # if "estaillades_sim2" not in sample:continue
        # if SSAporosityinfo.Publication[sample] != "Pe_0.0005": continue
        # if "Menke2017" not in SSAporosityinfo.Publication[sample]:continue
        reff = [float(ele) for ele in SSAporosityinfo.reff[sample][1:-1].split(",")]
        timesteps = [float(ele) for ele in SSAporosityinfo.Timestep[sample][1:-1].split(",")][1:] #in seconds
        if  SSAporosityinfo.Steps[sample] >= 50:
            n = 5
        else: n =2 
        averagedreff = moving_average(reff,n=n)
        averagedtime = moving_average(timesteps,n=n)
        changeiny = abs(np.diff(averagedreff))/averagedreff[0]
        changeinx = abs(np.diff(averagedtime))
        rateofchange = changeiny/changeinx
        
        index = check_convergence(rateofchange)
        if index == None: 
            print("no convergence for sample ", sample, " final reff: ",reff[-1])
        else:
            print(sample, "reff: ", np.mean(averagedreff[index:]), " at ", timesteps[index])
            if plot_:
                ax.plot(timesteps,reff,'-.',label=sample)
    if plot_:
        ax.set_ylim(0,1.2e-4)
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Effective Reaction Rate (reff) [mol/m2s]")
        ax.legend()
        fig.tight_layout()
        plt.show()
