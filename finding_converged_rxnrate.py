import os
import numpy as np
import pandas as pd

def check_convergence(roc, count,threshold=2):
    # print(count)
    found = False
    for ind,r in enumerate(roc):
        if found:
            print(r)
            break
        if r < threshold:
            count += 1
            if count < 5:
                found = check_convergence(roc[ind+1:],count)
            elif count >= 5: 
                print("converged")
                return True
                break
        else: 
            count = 0
            
            


reff_file = os.path.normpath(r'C:\Users\zkana\Documents\FlowHeterogeneity_and_Rxn\data\ket31_rxnrate.csv')
reff = pd.read_csv(reff_file,names=["time","r_eff"])


reff_diff = abs(np.diff(reff.r_eff))
time_diff = abs(np.diff(reff.time))
percent_change = (reff_diff/reff.r_eff.iloc[:-1])
rate_of_change = (percent_change/time_diff)*100

check_convergence(rate_of_change.values, count=0)

# print("final rate of change:", rateof_change.iloc[-1]*100, "%")