from inspect import CO_ITERABLE_COROUTINE
import os
import numpy as np
import pandas as pd

def check_convergence(roc, threshold=2):
    for ind,r in enumerate(roc):
        if all(roc[ind:] < threshold):
            # print("converged")
            return ind
        else:
            continue
            
# reff_file = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\data\ket31_rxnrate.csv')
directory = os.path.normpath(r'C:\Users\zkanavas\Documents\FlowHeterogeneity_and_Rxn\data\menke_2017')
filenames = ['ket36.csv','ket31.csv','est36.csv','est31.csv','port36.csv','port31.csv']

for file in filenames:
    print(" ")
    reff_file = directory + "/" + file
    reff = pd.read_csv(reff_file,names=["time","r_eff"])
    reff_diff = abs(np.diff(reff.r_eff))
    time_diff = abs(np.diff(reff.time))
    percent_change = (reff_diff/reff.r_eff.iloc[:-1])
    rate_of_change = (percent_change/time_diff)*100
    ind = check_convergence(rate_of_change.values)
    mean_reff = reff.r_eff[ind:].mean()
    print("sample: ", file[:-4], " converged reff: ", mean_reff, " time: ",reff.time[ind], " min")

directory = os.path.normpath(r'C:\Users\zkanavas\Documents\FlowHeterogeneity_and_Rxn\data\alkhulafi_2018')
filenames = ['HetALow.csv','HetAHigh.csv','HetBLow.csv','HetBHigh.csv']

for file in filenames:
    print(" ")
    reff_file = directory + "/" + file
    reff = pd.read_csv(reff_file,names=["time","r_eff"])
    reff_diff = abs(np.diff(reff.r_eff))
    time_diff = abs(np.diff(reff.time))
    percent_change = (reff_diff/reff.r_eff.iloc[:-1])
    rate_of_change = (percent_change/time_diff)*100
    ind = check_convergence(rate_of_change.values)
    mean_reff = reff.r_eff[ind:].mean()
    print("sample: ", file[:-4], " converged reff: ", mean_reff, " time: ",reff.time[ind], " min")

directory = os.path.normpath(r'C:\Users\zkanavas\Documents\FlowHeterogeneity_and_Rxn\data\nunes_2016')
filenames = ['estaillades.csv','ketton.csv','beadpack.csv']

for file in filenames:
    print(" ")
    reff_file = directory + "/" + file
    reff = pd.read_csv(reff_file,names=["time","r_eff"])
    reff_diff = abs(np.diff(reff.r_eff))
    time_diff = abs(np.diff(reff.time))
    percent_change = (reff_diff/reff.r_eff.iloc[:-1])
    rate_of_change = (percent_change/time_diff)*100
    ind = check_convergence(rate_of_change.values,threshold=1)
    mean_reff = reff.r_eff[ind:].mean()
    print("sample: ", file[:-4], " converged reff/rb: ", mean_reff, " time: ",reff.time[ind], " PV")

directory = os.path.normpath(r'C:\Users\zkanavas\Documents\FlowHeterogeneity_and_Rxn\data\alkhulafi_2019')
filenames = ['AH Calcite.csv','AH Dolomite.csv','AL Calcite.csv','AL Dolomite.csv','BH Calcite.csv','BH Dolomite.csv','BL Calcite.csv','BL Dolomite.csv']

#all are 90/10 dolomite/calcite

for file in filenames:
    print(" ")
    reff_file = directory + "/" + file
    reff = pd.read_csv(reff_file,names=["time","r_eff"])
    reff_diff = abs(np.diff(reff.r_eff))
    time_diff = abs(np.diff(reff.time))
    percent_change = (reff_diff/reff.r_eff.iloc[:-1])
    rate_of_change = (percent_change/time_diff)*100
    ind = check_convergence(rate_of_change.values)
    mean_reff = reff.r_eff[ind:].mean()
    print("sample: ", file[:-4], " converged reff: ", mean_reff, " time: ",reff.time[ind], " min")
