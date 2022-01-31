import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def compute_prediction(pc,PeV,beta0,beta1,beta2):
    y = np.exp(beta0 + beta1*pc+ beta2*PeV)
    return y



df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df.drop([3,9,14],axis=0,inplace=True)

y_obs = df.ratio

# y ~ 1+pc+Pe:Volhv
beta0 = -2.69
beta1 = -0.25
beta2 = 0.015
y_pred = compute_prediction(df.pc,df.Pe*df.Vol_hv,beta0,beta1,beta2)
r2 = r2_score(y_obs,y_pred)
print(r2)

#y ~1 + pc + Pe
beta0 = -2.244
beta1 = -0.346
beta2 = 7.8555e-4
y_pred = compute_prediction(df.pc,df.Pe,beta0,beta1,beta2)
r2 = r2_score(y_obs,y_pred)
print(r2)

