import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

samples = ["beadpack","ketton","estaillades"]

for samp in samples:
    for phase in ["_before.csv","_after.csv"]:
        vel_magnitude = pd.read_csv(samp+phase,header=0)
        mean = np.mean(np.exp(vel_magnitude.u))
        std = np.std(np.exp(vel_magnitude.u))
        cumulsum = np.cumsum(vel_magnitude.PDF.values)
        velmag_cdf = cumulsum/cumulsum[-1]

        #generate homogeneous (Gaussian) distribution
        normal = np.random.normal(loc=mean,scale = std, size=vel_magnitude.u.size)
        normal[normal<0]=0
        # pdf,norm_bins = np.histogram(normal,density=True,bins = 5)
        cumulsum = np.cumsum(normal)
        normal_cdf = cumulsum/cumulsum[-1]
        #calculate statistical distance between velocity distribution and normal distribution
        distance = stats.wasserstein_distance(vel_magnitude.PDF,normal)
        d2 = np.sum(np.abs(velmag_cdf-normal_cdf))
        print(samp,phase," wasserstein: ",distance)

for samp in samples:
    vel_magnitude_before = pd.read_csv(samp+"_before.csv",header=0)
    vel_magnitude_after = pd.read_csv(samp+"_after.csv",header=0)
    #calculate statistical distance between velocity distribution and normal distribution
    distance = stats.wasserstein_distance(vel_magnitude_before.PDF,vel_magnitude_after.PDF)
    print(samp," wasserstein: ",distance)