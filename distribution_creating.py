#distributions
# from scipy.stats import gamma,norm,wasserstein_distance,exponpow,lognorm
from scipy.special import gamma, gammainc, erf
import matplotlib.pyplot as plt
import numpy as np

#exponential, gamma, powerlaw

def exponential_distribution(mean,std,x):
    gamma = 1/mean
    exp_cdf = 1 - np.exp()**(-gamma*x)
    return exp_cdf

def gamma_distribution(mean,std,x):
    theta = std**2/mean
    k = mean/theta
    gamma_cdf = gammainc(k,(x/theta))/gamma(k)
    return gamma_cdf

#def power_distribution():

def normal_distribution(mean,std,x):
    normal_cdf = (1/2)*(1+erf((x-mean)/(std*(2)**(1/2))))

rng = np.random.default_rng()

#generate distribtions
# n = rng.normal(loc=2,scale=1,size=5000)
# g = rng.gamma(shape=1,scale=1,size=5000)
# log = rng.lognormal(size = 5000)
# #generate normal distributions
# gamma_n = rng.normal(loc=np.mean(g),scale=np.std(g),size=5000)
# log_n = rng.normal(loc=np.mean(log),scale=np.std(log),size=5000)

#rescaled to [0,100 range]
# g_rescaled = 100*(g - np.min(g))/np.ptp(g)
# log_rescaled = 100*(log-np.min(log))/np.ptp(log)
# gamma_n_rescaled = 100*(gamma_n-np.min(gamma_n))/np.ptp(gamma_n)
# log_n_rescaled = 100*(log_n-np.min(log_n))/np.ptp(log_n)

# distance = wasserstein_distance(g,gamma_n)
# print('gamma distance: ',distance)
# distance = wasserstein_distance(log,log_n)
# print('lognormal distance: ',distance)

#visualize distributions
#gamma
# fig,ax = plt.subplots()
# # bins = 10 ** np.linspace(np.log10(np.min(g_rescaled[g_rescaled != 0])), np.log10(np.max(g_rescaled)),num=100)
# num,bins = np.histogram(g,bins = 100) 
# # densities = num*np.diff(bins)
# pdf =num/np.sum(num)
# cdf = np.cumsum(pdf)
# ax.plot(bins[:-1],pdf, label = 'gamma', linewidth = 2)
# # bins = 10 ** np.linspace(np.log10(np.min(gamma_n_rescaled[gamma_n_rescaled != 0])), np.log10(np.max(gamma_n_rescaled)),num=100)
# num,normal_bins = np.histogram(gamma_n,bins = 100) 
# # densities = num*np.diff(normal_bins)
# pdf =num/np.sum(num)
# normal_cdf = np.cumsum(pdf)
# ax.plot(normal_bins[:-1],pdf, label = 'normal', linewidth = 2) 
# #general properties
# # ax.semilogx()
# ax.tick_params(axis='both',labelsize=14)
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('PDF',fontsize=15)
# ax.legend()
# plt.tight_layout()

#lognormal
# fig,ax = plt.subplots()
# # bins = 10 ** np.linspace(np.log10(np.min(log_rescaled[log_rescaled != 0])), np.log10(np.max(log_rescaled)),num=100)
# num,bins = np.histogram(log,density=True,bins = 100) 
# # densities = num*np.diff(bins)
# pdf =num/np.sum(num)
# cdf = np.cumsum(pdf)
# ax.plot(bins[:-1],cdf, label = 'lognormal', linewidth = 2)
# # bins = 10 ** np.linspace(np.log10(np.min(log_n_rescaled[log_n_rescaled != 0])), np.log10(np.max(log_n_rescaled)),num=100)
# num,normal_bins = np.histogram(log_n,density=True,bins = bins) 
# # densities = num*np.diff(normal_bins)
# pdf =num/np.sum(num)
# normal_cdf = np.cumsum(pdf)
# ax.plot(normal_bins[:-1],cdf, label = 'normal', linewidth = 2) 
# #general properties
# # ax.semilogx()
# ax.tick_params(axis='both',labelsize=14)
# ax.set_xlabel('x', fontsize=15)
# ax.set_ylabel('CDF',fontsize=15)
# ax.legend()
# plt.tight_layout()


# plt.show()

