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
# links = ['identity']
link = 'power' #identity, log, power, inverse
# for link in links:
if link == 'power': 
    numberofparams=16
    lb=0
elif link == 'identity':
    numberofparams = 16
    lb = -10
elif link == 'log' or link == 'inverse': 
    numberofparams=16
    lb = 0
else: print('link label invalid')

#set weight bounds
bounds = [(lb,10)for x in range(numberofparams)]

samples = df.index.tolist()
ratios = np.asarray([df.loc[sample]['ratio'] for sample in samples])
inputs =(df,samples,link,ratios)


randomstates = np.random.randint(2,4000,500)
# randomstates=[2102]
# r2s = []
# aics = []
# solutions = []
# non_zeros = []
solution_df = pd.DataFrame()
for count,randomstate in enumerate(randomstates):
    print('loop ',count)

    np.random.seed(randomstate)

    init = generate_initial_population(numberofparams,inputs)
    w = [0.,10, 0.,10,0.,0., 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    init = np.row_stack((init,np.asarray(w)))
    # init = np.identity(numberofparams)

    res = differential_evolution(calculate_rxn, bounds=bounds,
                                seed=1,mutation=(0.5,1),
                                recombination=0.9,strategy='best1bin',
                                atol=0,tol=0.01,polish=True,
                                init=init,
                                # constraints=nlc,
                                args=inputs)
    # print(res)
    w =res.x

    r2,aic = calculate_rxn(w,*inputs,returnr2=True)

    solution_gof = np.append(w,(r2,aic))
    solution = pd.DataFrame(solution_gof.reshape(1,-1))
    solution_df = solution_df.append(solution)
solution_df.to_csv('seedsens_power.csv',index=False)
        # r2s.append(r2)
        # aics.append(aic)
        # solutions.append(w)
        # non_zeros.append(no_zero)

    # data = {'seed':randomstates,'r2':r2s,'aic':aics,'solution':solutions,'numbernonzeros':non_zeros}
    # solution_df = pd.DataFrame(data=data)
    # solution_df.to_csv("modelopt_randseeds.csv")