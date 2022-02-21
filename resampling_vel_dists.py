import pandas as pd
import numpy as np

df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
df.drop("menke_ketton",inplace=True)

# samples = df.index
# normfreqsamples = ["AH","AL","BH","BL","Sil_HetA_High_Scan1","Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]

samp = "Sil_HetA_High_Scan1"
phase = "_before.csv"
vel_magnitude = pd.read_csv(samp+phase,header=None)

dx = [vel_magnitude[0][ind]-vel_magnitude[0][ind-1] for ind in range(1,len(vel_magnitude[0]))]
dx = np.insert(dx, 0,dx[0])

resampd_array = np.random.choice(vel_magnitude[0],p=(vel_magnitude[1]*dx))

mean = np.sum(vel_magnitude[0]*vel_magnitude[1]*dx)
std = ((np.sum((vel_magnitude[0]-mean)**2))/len(vel_magnitude[0]))**(1/2)

print(samp,phase,"sum of y-curve: ",np.sum(vel_magnitude[1]),"AUC: ", np.sum(dx*vel_magnitude[1])," mean: ",mean, " std: ",std)
