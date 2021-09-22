import pandas as pd
import numpy as np
import itertools
from scipy.optimize import differential_evolution,NonlinearConstraint,minimize
from optimizemodel import calculate_rxn,generate_initial_population
import csv

#import data
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)

#extract parameters
df.pc = df.pc/np.max(df.pc)
df.EMD = df.EMD/np.max(df.EMD)
df.Mm = df.Mm/np.max(df.Mm)
df.Pe = df.Pe/np.max(df.Pe)
df.Da = df.Da/np.max(df.Da)

#create individual & interaction terms
pc = df['pc'].values
EMD =df['EMD'].values
Mm = df['Mm'].values
Pe = df['Pe'].values
Da = df['Da'].values
pcEMD = pc*EMD
pcMm = pc*Mm
pcPe = pc*Pe
pcDa = pc*Da
EMDMm = EMD*Mm
EMDPe = EMD*Pe
EMDDa = EMD*Da
MmPe = Mm*Pe
MmDa = Mm*Da
PeDa = Pe*Da

#create variable list
variable_list = [pc,EMD,Mm,Pe,Da,pcEMD,pcMm,pcPe,pcDa,EMDMm,EMDPe,EMDDa,MmPe,MmDa,PeDa]
variable_names = ['pc','EMD','Mm','Pe','Da','pcEMD','pcMm','pcPe','pcDa','EMDMm','EMDPe','EMDDa','MmPe','MmDa','PeDa']
ratios = df['ratio'].values

# variable_list = np.arange(4)
# variable_names = ['0','1','2','3']

#build base model
# df = inputs[0]
# Samples = inputs[1]
def model(weights,*inputs):
    link = inputs[0][0]
    ratios = inputs[0][1]
    variables = inputs[0][2]
    variables = np.row_stack((variables, np.ones(len(variables))))
    # predictions = weights[0] + [w*v for w,v in zip(weights,variables)]
    predictions = weights*variables
    predictions = np.asarray([np.sum(row) for row in predictions])
    #link function
    if link == 'identity': predictions = predictions
    # elif link == 'log': predictions = [(1/alpha)*np.exp(-x/alpha) for x in predictions]
    elif link == 'log': predictions = [np.exp(-x) for x in predictions]
    elif link == 'inverse': predictions = [1/x for x in predictions]
    elif link == 'power': predictions = [x**(-1.5) for x in predictions]
    else: return 'link label invalid'
    mean_ratio = np.mean(ratios)

    r2 = 1 - (np.sum((predictions-ratios)**2))/(np.sum((ratios-mean_ratio)**2))
    # print(r2)
    log_likelihood = np.sum(-1*((ratios/predictions)+np.log(predictions)))
    nonzer0s = np.count_nonzero(weights)
    # if link == 'power': nonzer0s -= 1
    # print(log_likelihood)
    aic = 2*(nonzer0s) - 2*log_likelihood
    return aic

def run_and_save(numberofparams,lb,ub,link,variables,ratios,label,filename):
    initial_guess = [round(x,2) for x in np.random.uniform(lb, ub, size=(numberofparams))]
    bounds = [(lb,ub)for x in range(numberofparams)]
    inputs = [link,ratios,variables]
    res = minimize(model, initial_guess,bounds=bounds,args=inputs)
    res['label'] = label
    # keys = res.keys()
    with open(filename,'a',newline='') as csvfile:
        writer = csv.DictWriter(csvfile,fieldnames= res.keys())
        # writer.writeheader()
        # for data in res:
        writer.writerow(res)
    return res

filename = 'gradient_descent_on_combos.csv'


#1-term models
# for label,variable in zip(variable_names,variable_list): #obj_fun = w[0] + w[1]*variable
#     #define model
#     run_and_save(2,0,10,'power',variable,ratios,label,filename)

#2-term models
combos = itertools.combinations(variable_list,2)
combo_labels = itertools.combinations(variable_names,2)
for label,variables in zip(combo_labels,combos):
    run_and_save(3,0,10,'power',variables,ratios,label,filename)

#3-term models
combos = itertools.combinations(variable_list,3)
combo_labels = itertools.combinations(variable_names,3)
for label,variables in zip(combo_labels,combos):
    run_and_save(4,0,10,'power',variables,ratios,label,filename)
