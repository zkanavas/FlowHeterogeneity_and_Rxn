import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import os

def resample_vel_distribution(vel_magnitude):
    new_array = np.ones(1e6)
    xs = vel_magnitude[0]
    ys = vel_magnitude[1]
    
def dx_mean_std(vel_magnitude):
    dx = [vel_magnitude[0][ind]-vel_magnitude[0][ind-1] for ind in range(1,len(vel_magnitude[0]))]
    dx = np.insert(dx, 0,dx[0])
    # if any(dx < 0): print(samp,"before")
    auc = np.sum(dx*vel_magnitude[1])
    vel_magnitude[1] /= auc
    auc = np.sum(dx*vel_magnitude[1])
    mean = np.sum(vel_magnitude[0]*vel_magnitude[1]*dx)#/np.sum(dx*vel_magnitude[1])
    std = ((np.sum((vel_magnitude[0]-mean)**2))/len(vel_magnitude[0]))**(1/2)
    
    return dx,auc, mean, std

def mean_velocity(filename,datatype):
    vel_magnitude = np.fromfile(filename, dtype=np.dtype(datatype)) 
    # vel_magnitude = vel_magnitude.reshape(imagesize)
    mean = np.mean(vel_magnitude[vel_magnitude != 0])
    return mean

def rescale_to_mean(samp,before=True, after=True,vel_magnitude_before=[],vel_magnitude_after=[]):
    if samp == "AH" :
        if before == True:
            vel_magnitude_before[0]*=0.113
        if after == True:
            vel_magnitude_after[0]*=0.214
    elif samp == "AL":
        if before == True:
            vel_magnitude_before[0]*=0.099
        if after == True:
            vel_magnitude_after[0]*=0.153
    elif samp == "BH":
        if before == True:
            vel_magnitude_before[0]*=0.080
        if after == True:
            vel_magnitude_after[0]*=0.180
    elif samp == "BL":
        if before == True:
            vel_magnitude_before[0]*=0.079
        if after == True:
            vel_magnitude_after[0]*=0.130
    elif samp == "Sil_HetA_High_Scan1":
        if before == True:
            vel_magnitude_before[0]*=0.204
        if after == True:
            vel_magnitude_after[0]*=0.264
    elif samp == "Sil_HetA_Low_Scan1":
        if before == True:
            vel_magnitude_before[0]*=0.209
        if after == True:
            vel_magnitude_after[0]*=0.236
    elif samp == "Sil_HetB_High_Scan1":
        if before == True:
            vel_magnitude_before[0]*=0.184
        if after == True:
            vel_magnitude_after[0]*=0.266
    elif samp == "Sil_HetB_Low_Scan1":
        if before == True:
            vel_magnitude_before[0]*=0.182
        if after == True:
            vel_magnitude_after[0]*=0.198
    
    return vel_magnitude_before,vel_magnitude_after

def resample_pdf(vel_magnitude,dx):
    # probs = (vel_magnitude[1]*dx)
    # vel_magnitude[1] /= np.sum(vel_magnitude[1])
    # resampled_pdf = np.random.choice(vel_magnitude[0],size=100000000,p=vel_magnitude[1])

    resampled_pdf = np.random.choice(vel_magnitude[0],size=100000000,p=(vel_magnitude[1]*dx))
    return resampled_pdf

plot_ = True
calc_metric = False
resample_only = False
# samples = ["menke_2017_est","menke_2017_ket","menke_2017_ket36","menke_2017_portland"]
# samples = ["Sil_HetA_High","Sil_HetA_Low","Sil_HetB_High","Sil_HetB_Low"]
df = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
df.drop(["menke_ketton","geometry0000"],inplace=True)
samples = df.index
# samples = ["AH","AL","BH","BL"]
normfreqsamples = ["AH","AL","BH","BL","Sil_HetA_High_Scan1","Sil_HetA_Low_Scan1","Sil_HetB_High_Scan1","Sil_HetB_Low_Scan1"]

# directory = os.path.normpath(r'F:\FlowHet_RxnDist')
# datatype = 'float32'


# down_dir = r"C:\Users\zkanavas\Downloads"
# down_dir = r"C:\Users\zkana\Downloads"
# for samp in samples:
#     vel_magnitude_before1 = pd.read_csv(samp+"_before.csv",header=None)
#     vel_magnitude_before2 = pd.read_csv(down_dir+"/"+samp+"_before.csv",header=None)
#     # vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=None)

#     dx,auc1, mean, std=dx_mean_std(vel_magnitude_before1)

#     dx,auc2, mean, std=dx_mean_std(vel_magnitude_before2)
#     print(samp,auc1,auc2)
#     fig,ax = plt.subplots()
#     # ax.plot(vel_magnitude_before1[0],vel_magnitude_before1[1],label="simple")
#     ax.plot(vel_magnitude_before2[0],vel_magnitude_before2[1],label="detailed",marker="o")
#     ax.legend()
#     ax.semilogx()
#     ax.set_title(samp)
# plt.show()
    # print(dx_mean_std(vel_magnitude_before))


if resample_only == True:
    for samp in samples:
        if samp not in ["AH"]: continue 
        vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=None)
        vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=None)
        if samp in normfreqsamples: 
            vel_magnitude_before,vel_magnitude_after = rescale_to_mean(samp,vel_magnitude_before=vel_magnitude_before,vel_magnitude_after=vel_magnitude_after)
        # else: continue
        dx_before,auc_before, mean_before, std_before=dx_mean_std(vel_magnitude_before)
        dx_after,auc_after, mean_after, std_after=dx_mean_std(vel_magnitude_after)
        print(samp," before: ",auc_before, mean_before,std_before)
        print(samp," after: ", auc_after, mean_after,std_after)

        resampled_before = resample_pdf(vel_magnitude_before,dx_before)
        resampled_after = resample_pdf(vel_magnitude_after,dx_after)

        fig,ax = plt.subplots()
        bins=np.logspace(np.log10(np.min(resampled_after)),np.log10(np.max(resampled_after)), 50)
        ax.hist(resampled_after, bins= bins,density=True, alpha=0.6,label="after")
        bins=np.logspace(np.log10(np.min(resampled_before)),np.log10(np.max(resampled_before)), 50)
        ax.hist(resampled_before, bins= bins,density=True, alpha= 0.6,label="before")
        # ax.plot(vel_magnitude_after[0],vel_magnitude_after[1],label="observed")
        ax.semilogx()
        ax.legend()
        ax.set_xlabel("U/Uave")
        ax.set_ylabel("PDF")
        ax.set_title(samp)
plt.show()

# before_distances = []
# after_distances = []
# before_after_distances = []
all_dist = []
if calc_metric == True:
    for samp in samples:
        vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=None)
        vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=None)
        # if samp not in ["BL"]:continue
        if samp in normfreqsamples: 
            vel_magnitude_before,vel_magnitude_after = rescale_to_mean(samp,vel_magnitude_before=vel_magnitude_before,vel_magnitude_after=vel_magnitude_after)
      
        dx_before,auc_before, mean_before, std_before=dx_mean_std(vel_magnitude_before)
        dx_after,auc_after, mean_after, std_after=dx_mean_std(vel_magnitude_after)
        # print(samp," before: ",auc_before, mean_before,std_before)
        # print(samp," after: ", auc_after, mean_after,std_after)

        resampled_before = resample_pdf(vel_magnitude_before,dx_before)
        resampled_after = resample_pdf(vel_magnitude_after,dx_after)

        distance_before_after = stats.wasserstein_distance(resampled_before,resampled_after)

        #rescale to mean so we can make lognormal dist
        # resampled_before /= np.mean(resampled_before)
        # resampled_after /= np.mean(resampled_after)

        mean_before = np.mean(resampled_before)
        mean_after = np.mean(resampled_after)
        std_before = np.std(resampled_before)
        std_after = np.std(resampled_after)
        
        lognormal_before = np.random.lognormal(mean=mean_before,sigma=std_before,size=1000000)
        distance_before = stats.wasserstein_distance(resampled_before,lognormal_before)

        lognormal_after = np.random.lognormal(mean=mean_after,sigma=std_after,size=1000000)
        distance_after = stats.wasserstein_distance(resampled_after,lognormal_after)

        distance_before_after_rescaled = stats.wasserstein_distance(resampled_before,resampled_after)

        # print(samp, "before AUC: ", np.sum(dx_before*vel_magnitude_before[1]))
        # print(samp, "after AUC: ", np.sum(dx_after*vel_magnitude_after[1]))

            # print(samp,phase,"AUC: ", np.sum(dx*vel_magnitude[1])," mean: ",mean, " std: ",std)
            # print(samp,phase,"log-normal wasserstein: ",distance)
        # before_distances.append(distance_before)
        # after_distances.append(distance_after)
        # before_after_distances.append(distance_before_after)
        all_dist.append([samp,distance_before,distance_after,distance_before_after,distance_before_after_rescaled])
        # fig,ax = plt.subplots()
        # ax.plot(vel_magnitude_before[0],vel_magnitude_before[1])
        # ax.plot(vel_magnitude_after[0],vel_magnitude_after[1])
        # ax.semilogx()
        # ax.set_title(samp)

    # print(samp,distances[1]-distances[0])
    # dist.append(distances)
    distances = pd.DataFrame(data=all_dist,columns=("Sample_Names","before","after","before_after","before_after_rescaled"))
# all_dist = []
# for ind, samp in enumerate(samples):
#     all_dist.append(samp,before_distances[ind],after_distances[ind],before_after_distances[ind])
# distances.to_csv("EMD_distancess_raw.csv")

# print(before_distances)
# print(after_distances)

distances = pd.read_csv("EMD_distancess_raw.csv",header=0,index_col="Sample_Names")
# distances.drop("BH",inplace=True)
samples = distances.index

plt.show()
if plot_ == True:
    size_min = 10
    size_max = 250
    scaled_distances = ((distances.before_after_rescaled - np.min(distances.before_after_rescaled))/np.ptp(distances.before_after_rescaled))*(size_max-size_min)+size_min
    # behavior = ["red" if beh == "uniform" else "blue" for beh in df.behavior]

    fig,ax = plt.subplots()
    # ax.grid(visible=True,zorder=1)
    # for ind,samp in enumerate(samples):
    ind = 0
    for samp in samples:
        # if samp == "BH": print(samp)
        behavior = df.behavior[samp]
        if behavior == "uniform": color = "red" 
        elif behavior == "wormhole": color ="blue" 
        elif behavior == "compact": color = "green"
        # ax.scatter(df.ratio[samp],distances.after[samp], c = color,s=scaled_distances[ind],label=samp,zorder=0)
        # ax.scatter(distances.before[samp],distances.after[samp], c = color,s=scaled_distances[ind], label=samp)
        ax.scatter(df.ratio[samp],distances.before_after_rescaled[samp], c = color, label=samp)
        ind += 1
    # ax.set_ylim(-1e200,1e3)
    # ax.plot([0,np.max(distances.before)],[0,np.max(distances.before)],'k-')
    # ax.loglog()
    ax.semilogy()
    # ax.set_xlabel("before-LogNormal EMD",fontsize=15)
    # ax.set_ylabel("after-LogNormal EMD",fontsize=15)
    # ax.set_ylabel("before-LogNormal EMD",fontsize=15)
    ax.set_xlabel("Rxn Ratio",fontsize=15)
    ax.set_ylabel("before-after EMD",fontsize=15)
    ax.tick_params(labelsize=12)
    
    # ax.legend()

    # fig,ax = plt.subplots()
    # ind = 0
    # # for ind,samp in enumerate(samples):
    # for samp in samples:
    #     # if samp in normfreqsamples: continue
    #     behavior = df.behavior[samp]
    #     if behavior == "uniform": color = "red" 
    #     elif behavior == "wormhole": color ="blue" 
    #     elif behavior == "compact": color = "green"
    #     ax.scatter(before_distances[ind],after_distances[ind],s=scaled_distances[ind], c = color,label=samp)
    #     ind += 1
    fig.tight_layout()
    plt.show()

# for samp in samples:
    # vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=0)
    # vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=0)
#     #calculate statistical distance between velocity distribution and normal distribution
#     distance = stats.wasserstein_distance(vel_magnitude_before[1],vel_magnitude_after[1]])
    # print(samp," wasserstein: ",distance)