import numpy as np
from numpy import random
from optimizemodel import calculate_rxn,generate_initial_population
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


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

# randomstates = np.random.randint(2,4000,500)
randomstate=[101]#[2102]
np.random.seed(randomstate)

samples = df.index.tolist()
ratios = np.asarray([df.loc[sample]['ratio'] for sample in samples])
inputs =(df,samples,link,ratios)    
init = generate_initial_population(numberofparams,inputs)
w = [0.,10, 0.,10,0.,0., 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
init = np.row_stack((init,np.asarray(w)))
# r2s = []
# aics = []
# solutions = []
# non_zeros = []

kfold = KFold(12,shuffle=True,random_state=1)
fold = 1
powers=[]
for train_index, test_index in kfold.split(samples):
    
    trainingset, validatingset = np.array(samples)[train_index], np.array(samples)[test_index]
    
    ratios = np.asarray([df.loc[sample]['ratio'] for sample in trainingset])
    
    inputs =(df,trainingset,link,ratios)
    
    # init = generate_initial_population(numberofparams,inputs)

    res = differential_evolution(calculate_rxn, bounds=bounds,
                                seed=1,mutation=(0.5,1),
                                recombination=0.9,strategy='best1bin',
                                atol=0,tol=0.01,polish=True,
                                init=init,
                                # constraints=nlc,
                                args=inputs)
    # print(res)
    
    #check results on whole set
    ratios = np.asarray([df.loc[sample]['ratio'] for sample in samples])
    inputs =(df,samples,link,ratios)

    w =res.x
    # powers.append(w[16])
    no_zero = np.count_nonzero(w)

    r2,aic = calculate_rxn(w,*inputs,returnr2=True)

    print(link, str(fold))
    print(w)
    print('r2: ', r2, ', aic: ',aic, ', nonzeros: ',no_zero)
    fold +=1
# fig, ax = plt.subplots()
# ax.hist(powers,bins=3,ec='darkblue')
# ax.set_xlabel('Power')
# ax.set_ylabel('Count')
# plt.show()