import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

# samples = ["menke_2017_est","menke_2017_ket","menke_2017_ket36","menke_2017_portland"]
# samples = ["Sil_HetA_High","Sil_HetA_Low","Sil_HetB_High","Sil_HetB_Low"]
df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
df.drop("menke_ketton",inplace=True)
samples = df.index
normfreqsamples = ["AH","AL","BH","BL","Sil_HetA_High_Scan1","Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]

fig,ax = plt.subplots()

for samp in samples:
    if samp in normfreqsamples:continue
    distances = []
    for phase in ["_before.csv","_after.csv"]:
        vel_magnitude = pd.read_csv(samp+phase,header=None)
        dx = [vel_magnitude[0][ind]-vel_magnitude[0][ind-1] for ind in range(1,len(vel_magnitude[0]))]
        dx = np.insert(dx, 0,dx[0])
        if samp in normfreqsamples:
            mean = np.sum(vel_magnitude[0]*vel_magnitude[1])/np.sum(vel_magnitude[1]*dx)
        else:
            mean = np.sum(vel_magnitude[0]*vel_magnitude[1])/np.sum(vel_magnitude[1])
        std = ((np.sum((vel_magnitude[0]-mean)**2))/len(vel_magnitude[0]))**(1/2)
        # print(samp,phase,mean,std)

#         cumulsum = np.cumsum(vel_magnitude.PDF.values)
#         velmag_cdf = cumulsum/cumulsum[-1]

        #generate homogeneous (Gaussian) distribution
        normal = np.random.normal(loc=mean,scale = std, size=vel_magnitude[1].size)
        lognormal = np.random.lognormal(mean=mean,sigma=std,size=vel_magnitude[1].size)
        # normal[normal<0]=0
        # pdf,norm_bins = np.histogram(normal,density=True,bins = 5)
        # cumulsum = np.cumsum(normal)
        # normal_cdf = cumulsum/cumulsum[-1]
        #calculate statistical distance between velocity distribution and normal distribution
        distance = stats.wasserstein_distance(vel_magnitude[1],normal)
        # d2 = np.sum(np.abs(velmag_cdf-normal_cdf))
        # print(samp,phase,"normal wasserstein: ",distance)

        # distance = stats.wasserstein_distance(vel_magnitude[1],lognormal)
        # print(samp,phase,"log-normal wasserstein: ",distance)
        distances.append(distance)
    # print(samp,distances[1]-distances[0])
    ax.scatter(distances[0],distances[1],label=samp)
    # ax.scatter(df.ratio[samp],distances[1]-distances[0],label=samp)
# ax.set_ylim(-1e200,1e3)
# ax.semilogy()
ax.legend()
plt.show()

# for samp in samples:
#     vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=0)
#     vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=0)
#     #calculate statistical distance between velocity distribution and normal distribution
#     distance = stats.wasserstein_distance(vel_magnitude_before.PDF,vel_magnitude_after.PDF)
#     print(samp," wasserstein: ",distance)