import numpy as np
from numpy import random
from optimizemodel import calculate_rxn,generate_initial_population
import pandas as pd
from scipy.optimize import differential_evolution

# #best so far
# aic = (9/18) - 0.8935

# for i in range(1,18):
#     r2 = -1*aic+ (i/18)
# randomstate = 2021

#import data
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
# df.drop(['fracturedB','Sil_HetA_High_Scan1','Sil_HetA_Low_Scan1','Sil_HetB_High_Scan1','Sil_HetB_Low_Scan1'],inplace=True)
df.pc = df.pc/np.max(df.pc)
# df.EMD = np.log(df.EMD)
df.EMD = df.EMD/np.max(df.EMD)
# df.Mm = (np.exp(df.Mm)/(1+np.exp(df.Mm)))
df.Mm = df.Mm/np.max(df.Mm)
df.Pe = df.Pe/np.max(df.Pe)
df.Da = df.Da/np.max(df.Da)

#set link
link = 'exp' #identity
if link == 'exp': 
    numberofparams=17
    lb = 0
else: 
    numberofparams=16
    lb = -1

#set weight bounds
bounds = [(lb,1)for x in range(numberofparams)]

samples = df.index.tolist()
inputs =(df,samples,link)

randomstates = np.random.randint(2,4000,500)
randomstates=[2102]
r2s = []
aics = []
solutions = []
non_zeros = []
for count,randomstate in enumerate(randomstates):
    print('loop ',count)

    np.random.seed(randomstate)

    init = generate_initial_population(numberofparams,inputs)

    res = differential_evolution(calculate_rxn, bounds=bounds,
                                seed=1,mutation=(0.5,1),
                                recombination=0.9,strategy='best1bin',
                                atol=0,tol=0.01,polish=True,
                                init=init,
                                # constraints=nlc,
                                args=inputs)

    w =res.x
    no_zero = np.count_nonzero(w)
    r2,aic = calculate_rxn(w,*inputs,returnr2=True)
    print(w)
    print('r2: ', r2, ', aic: ',aic, ', nonzeros: ',no_zero)
    
    # r2s.append(r2)
    # aics.append(aic)
    # solutions.append(w)
    # non_zeros.append(no_zero)

# data = {'seed':randomstates,'r2':r2s,'aic':aics,'solution':solutions,'numbernonzeros':non_zeros}
# solution_df = pd.DataFrame(data=data)
# solution_df.to_csv("modelopt_randseeds.csv")