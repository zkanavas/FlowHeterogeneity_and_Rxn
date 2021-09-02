#standardize
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)

# heterogeneities = ['pc','Mm','EMD']
heterogeneity = ['EMD']

#what I want to predict
rxnratio = df['ratio'].values.tolist()

#attributes
x = df['Pe'].values.tolist()
y = df['Da'].values.tolist()

# for heterogeneity in heterogeneities:
z = df[heterogeneity].values.tolist()

X = np.column_stack([x,y,z])

scaler = StandardScaler()
scaler.fit(X)
attributes = scaler.transform(X)
print(np.mean(attributes[:,0]),np.std(attributes))

# fig,ax = plt.subplots()
# ax.scatter(z,attributes)
# plt.show()