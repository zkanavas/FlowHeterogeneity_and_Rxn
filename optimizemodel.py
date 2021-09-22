#assess different model combinations
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution,NonlinearConstraint

def calculate_rxn(weights,*inputs,returnr2=False):
    df = inputs[0]
    Samples = inputs[1]
    link = inputs[2]
    ratios = inputs[3]

    #assign weights
    w0 = weights[0]
    w_pc = weights[1]
    w_EMD = weights[2]
    w_Mm = weights[3]
    w_Pe = weights[4]
    w_Da = weights[5]
    w_pcEMD = weights[6]
    w_pcMm = weights[7]
    w_pcPe = weights[8]
    w_pcDa = weights[9]
    w_EMDMm = weights[10]
    w_EMDPe = weights[11]
    w_EMDDa = weights[12]
    w_MmPe = weights[13]
    w_MmDa = weights[14]
    w_PeDa = weights[15]
    # if link != 'identity' and link != 'log':
    # if link == 'power':
    #     alpha = weights[16]    
    
    predictions = []
    for sample in Samples:
        #attributes: pc, EMD, Mm, Pe, Da
        pc = df.loc[sample]['pc']
        EMD =df.loc[sample]['EMD']
        Mm = df.loc[sample]['Mm']
        Pe = df.loc[sample]['Pe']
        Da = df.loc[sample]['Da']

        #responses: behavior, rxn ratio
        # behavior = df.loc[sample]['behavior']
        # rxnratio = df.loc[sample]['ratio']

        #create model
        response = w0 + w_pc*pc + w_EMD*EMD + w_Mm*Mm + w_Pe*Pe + w_Da*Da + w_pcEMD*(pc*EMD) + w_pcMm*(pc*Mm) + w_pcPe*(pc*Pe) + w_pcDa*(pc*Da) + w_EMDMm*(EMD*Mm) + w_EMDPe*(EMD*Pe) + w_EMDDa*(EMD*Da) + w_MmPe*(Mm*Pe) + w_MmDa*(Mm*Da) + w_PeDa*(Pe*Da)
        predictions.append(response)

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
    aic = 2*(nonzer0s) - 2*log_likelihood
    # if r2 > 0:
    #     aic = 2*(np.count_nonzero(weights)) - 2*np.log(r2)
    # else: aic = 1000
    if returnr2 == False: return aic#r2*-1
    elif returnr2 == True: 
        return r2,aic

def generate_uniform(numberofparams):
    uniform = np.zeros(numberofparams)
    uniform = [round(x,2) for x in np.random.uniform(0, 10, size=(numberofparams))]
    for x in np.random.choice(numberofparams,np.random.randint(7,numberofparams),replace=False): 
        uniform[x] = 0
    if numberofparams == 17:
        uniform[16] = round(np.random.uniform(1e-2,10),2)
        # uniform[17] = round(np.random.uniform(0,1),2)
    return uniform

def generate_initial_population(numberofparams,inputs):
    print('initializing...')
    candidates = []
    stop = False
    while stop == False:
        w = generate_uniform(numberofparams)
        r2,aic = calculate_rxn(w,*inputs,returnr2=True)
        if r2 > 0:
            candidates.append(w)
            print(np.count_nonzero(w)) 
            if len(candidates) == numberofparams:
                stop = True
                print('done initializing.')
    return candidates


# #import data
# df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
# # df.drop(['fracturedB','Sil_HetA_High_Scan1','Sil_HetA_Low_Scan1','Sil_HetB_High_Scan1','Sil_HetB_Low_Scan1'],inplace=True)
# df.pc = df.pc/np.max(df.pc)
# # df.EMD = np.log(df.EMD)
# df.EMD = df.EMD/np.max(df.EMD)
# # df.Mm = (np.exp(df.Mm)/(1+np.exp(df.Mm)))
# df.Mm = df.Mm/np.max(df.Mm)
# df.Pe = df.Pe/np.max(df.Pe)
# df.Da = df.Da/np.max(df.Da)

# numberofparams=18

# init = np.identity(numberofparams)

# #set weight bounds
# bounds = [(0,1)for x in range(numberofparams)]

# samples = df.index.tolist()
# inputs =(df,samples)

# adding contraint, requires the number of non-zero attributes in model to be less than or equal to 3
# def constr(w):
#     return np.count_nonzero(w)
# nlc = NonlinearConstraint(constr,1,18) #number of nonzero w's must be less than ...
# res = differential_evolution(calculate_rxn, bounds=bounds,
#                             seed=1,mutation=(0.5,1),
#                             recombination=0.9,strategy='best1bin',
#                             atol=0,tol=0.01,polish=True,
#                             init=init,
#                             # constraints=nlc,
#                             args=inputs)
# print(res)

# w =res.x
# r2 = calculate_rxn(w,*inputs,returnr2=True)
# print('r2 = ', r2)