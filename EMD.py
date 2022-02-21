import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def resample_vel_distribution(vel_magnitude):
    new_array = np.ones(1e6)
    xs = vel_magnitude[0]
    ys = vel_magnitude[1]
    
def dx_mean_std(vel_magnitude):
    dx = [vel_magnitude[0][ind]-vel_magnitude[0][ind-1] for ind in range(1,len(vel_magnitude[0]))]
    dx = np.insert(dx, 0,dx[0])
    # if any(dx < 0): print(samp,"before")

    mean = np.sum(vel_magnitude[0]*vel_magnitude[1]*dx)/np.sum(dx*vel_magnitude[1])
    std = ((np.sum((vel_magnitude[0]-mean)**2))/len(vel_magnitude[0]))**(1/2)
    AUC = np.sum(dx*vel_magnitude[1])
    return dx,AUC, mean, std

plot_ = False
calc_metric = False
# samples = ["menke_2017_est","menke_2017_ket","menke_2017_ket36","menke_2017_portland"]
# samples = ["Sil_HetA_High","Sil_HetA_Low","Sil_HetB_High","Sil_HetB_Low"]
df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
df.drop(["menke_ketton","geometry0000"],inplace=True)
samples = df.index
samples = ["AH","AL","BH","BL"]
normfreqsamples = ["AH","AL","BH","BL","Sil_HetA_High_Scan1","Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]

down_dir = r"C:\Users\zkanavas\Downloads"
for samp in samples:
    vel_magnitude_before1 = pd.read_csv(samp+"_before.csv",header=None)
    vel_magnitude_before2 = pd.read_csv(down_dir+"/"+samp+"_before.csv",header=None)
    # vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=None)
    # if samp in normfreqsamples:
    #     if samp == "AH" :
    #         vel_magnitude_before[0]*=0.113
    #         # vel_magnitude_after[0]*=0.214
    #     elif samp == "AL":
    #         vel_magnitude_before[0]*=0.099
    #         # vel_magnitude_after[0]*=0.153
    #     elif samp == "BH":
    #         vel_magnitude_before[0]*=0.080
    #         # vel_magnitude_after[0]*=0.180
    #     elif samp == "BL":
    #         vel_magnitude_before[0]*=0.079
            # vel_magnitude_after[0]*=0.130
    fig,ax = plt.subplots()
    ax.plot(vel_magnitude_before1[0],vel_magnitude_before1[1])
    ax.plot(vel_magnitude_before2[0],vel_magnitude_before2[1])
    ax.semilogx()
    ax.set_title(samp)
plt.show()
    # print(dx_mean_std(vel_magnitude_before))


before_distances = []
after_distances = []
before_after_distances = []
if calc_metric == True:
    for samp in samples:
        if samp == "AL":
            print('check')#continue
        vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=None)
        vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=None)
        if samp in normfreqsamples:
            if samp == "AH" :
                vel_magnitude_before[0]*=0.113
                vel_magnitude_after[0]*=0.214
            elif samp == "AL":
                vel_magnitude_before[0]*=0.099
                vel_magnitude_after[0]*=0.153
            elif samp == "BH":
                vel_magnitude_before[0]*=0.080
                vel_magnitude_after[0]*=0.180
            elif samp == "BL":
                vel_magnitude_before[0]*=0.079
                vel_magnitude_after[0]*=0.130
            elif samp == "Sil_HetA_High_Scan1":
                vel_magnitude_before[0]*=0.204
                vel_magnitude_after[0]*=0.264
            elif samp == "Sil_HetA_Low_Scan1":
                vel_magnitude_before[0]*=0.209
                vel_magnitude_after[0]*=0.236
            elif samp == "Sil_HetB_High_Scan1":
                vel_magnitude_before[0]*=0.184
                vel_magnitude_after[0]*=0.266
            elif samp == "Sil_HetB_Low_Scan1":
                vel_magnitude_before[0]*=0.182
                vel_magnitude_after[0]*=0.198

        distances = []
        
        dx_before = [vel_magnitude_before[0][ind]-vel_magnitude_before[0][ind-1] for ind in range(1,len(vel_magnitude_before[0]))]
        dx_before = np.insert(dx_before, 0,dx_before[0])
        if any(dx_before < 0): print(samp,"before")

        mean_before = np.sum(vel_magnitude_before[0]*vel_magnitude_before[1]*dx_before)/np.sum(dx_before*vel_magnitude_before[1])
        std_before = ((np.sum((vel_magnitude_before[0]-mean_before)**2))/len(vel_magnitude_before[0]))**(1/2)
        lognormal_before = np.random.lognormal(mean=mean_before,sigma=std_before,size=vel_magnitude_before[1].size)
        
        distance_before = stats.wasserstein_distance(vel_magnitude_before[1],lognormal_before)

        dx_after = [vel_magnitude_after[0][ind]-vel_magnitude_after[0][ind-1] for ind in range(1,len(vel_magnitude_after[0]))]
        dx_after = np.insert(dx_after, 0,dx_after[0])
        if any(dx_after < 0): print(samp,"after")

        mean_after = np.sum(vel_magnitude_after[0]*vel_magnitude_after[1]*dx_after)/np.sum(dx_after*vel_magnitude_after[1])
        std_after = ((np.sum((vel_magnitude_after[0]-mean_after)**2))/len(vel_magnitude_after[0]))**(1/2)
        lognormal_after = np.random.lognormal(mean=mean_after,sigma=std_after,size=vel_magnitude_after[1].size)
        distance_after = stats.wasserstein_distance(vel_magnitude_after[1],lognormal_after)

        distance_before_after = stats.wasserstein_distance(vel_magnitude_before[1],vel_magnitude_after[1])

        print(samp, "before AUC: ", np.sum(dx_before*vel_magnitude_before[1]))
        print(samp, "after AUC: ", np.sum(dx_after*vel_magnitude_after[1]))

            # print(samp,phase,"AUC: ", np.sum(dx*vel_magnitude[1])," mean: ",mean, " std: ",std)
            # print(samp,phase,"log-normal wasserstein: ",distance)
        before_distances.append(distance_before)
        after_distances.append(distance_after)
        before_after_distances.append(distance_before_after)

    # print(samp,distances[1]-distances[0])
    # dist.append(distances)

if plot_ == True:
    size_min = 10
    size_max = 1000
    scaled_distances = ((before_after_distances - np.min(before_after_distances))/np.ptp(before_after_distances))*(size_max-size_min)+size_min
    behavior = ["red" if beh == "uniform" else "blue" for beh in df.behavior]

    fig,ax = plt.subplots()
    for ind,samp in enumerate(samples):
        ax.scatter(df.ratio[samp],before_distances[ind]-after_distances[ind], c = behavior[ind], label=samp)
    # ax.set_ylim(-1e200,1e3)
    # ax.loglog()
    ax.semilogy()
    # ax.legend()

    fig,ax = plt.subplots()
    for ind,samp in enumerate(samples):
        ax.scatter(before_distances[ind],after_distances[ind],s=scaled_distances[ind], c = behavior[ind],label=samp)
    plt.show()

# for samp in samples:
    # vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=0)
    # vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=0)
#     #calculate statistical distance between velocity distribution and normal distribution
#     distance = stats.wasserstein_distance(vel_magnitude_before[1],vel_magnitude_after[1]])
    # print(samp," wasserstein: ",distance)