#distributions
# from scipy.stats import wasserstein_distance#,powerlaw,gamma,norm,exp
import scipy.stats as stats
from scipy.special import gamma, gammainc, erf
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.metrics as metrics
import pandas as pd
import random

plot = True

#exponential, gamma, powerlaw
rng = np.random.default_rng(12345)
def exponential_distribution(mean,std,x):
    lambda_ = 1/mean
    exp_cdf = 1 - math.e**(-lambda_*x)
    exp_cdf_func = stats.expon.cdf(x,scale = mean)
    exp_pdf = stats.expon.pdf(x,scale = mean)
    if all(np.logical_or(exp_cdf - 1e-5 <= exp_cdf_func, exp_cdf + 1e-5 >= exp_cdf_func)):
        return exp_cdf_func,exp_pdf
    else: return print('error in exponential distribution')

def gamma_distribution(mean,std,x):
    theta = std**2/mean
    k = mean/theta
    gamma_cdf = gammainc(k,(x/theta))/gamma(k)
    gamma_cdf_func = stats.gamma.cdf(x,k,scale=theta)
    gamma_pdf = stats.gamma.pdf(x,k,scale=theta)
    if all(np.logical_or(gamma_cdf - 1e-5 <= gamma_cdf_func, gamma_cdf + 1e-5 >= gamma_cdf_func)):
        return gamma_cdf_func, gamma_pdf
    else: return print('error in gamma distribution')

def power_distribution(mean, std, x): #using pareto distribution
    xm = 1e-10
    #alpha must be > 1 for well-defined mean and > 3 for finite variance
    alpha = ((3/xm)+((9+(8*xm)+(4*mean*xm**2))/((std**2)*(xm**2)))**(1/2))/(2/xm)
    if alpha < 3:
        return print('error! alpha is less than 3')
    powerlaw_cdf = 1 - (xm/x)**alpha
    powerlaw_cdf_func = stats.pareto.cdf(x,alpha,scale=xm)
    powerlaw_pdf = stats.pareto.pdf(x,alpha,scale=xm)
    if all(np.logical_or(powerlaw_cdf - 1e-5 <= powerlaw_cdf_func, powerlaw_cdf + 1e-5 >= powerlaw_cdf_func)):
        return powerlaw_cdf_func, powerlaw_pdf
    else: return print('error in powerlaw distribution')

def normal_distribution(mean,std,x):
    normal_cdf = (1/2)*(1+erf((x-mean)/(std*(2)**(1/2))))
    normal_cdf_func = stats.norm.cdf(x,loc=mean,scale=std)
    normal_pdf = stats.norm.pdf(x,loc=mean,scale=std)
    if all(np.logical_or(normal_cdf - 1e-5 <= normal_cdf_func, normal_cdf + 1e-5 >= normal_cdf_func)):
        return normal_cdf_func, normal_pdf
    else: return print('error in normal distribution')   

#note, when mean and std are 1, exp == gamma
x = np.linspace(1e-10,1,1000000)
# mu = np.linspace(1,5,10)
# sigma = np.linspace(.9,4.9,10)
mu = rng.random(10000)*10
# sigma = rng.integers(1,10,10)
mu = [3.79e-3]
sigma = np.linspace(1e-5,1.39,50)
sigma = [3e-3]
for mean,std in zip(mu,sigma):
    # std = random.uniform(0, (1/3)*(4*(mean)+17)**(1/2)) #required to make sure alpha > 3 for powerlaw distribution
    exp_cdf,exp_pdf = exponential_distribution(mean,std,x)
    gamma_cdf,gamma_pdf = gamma_distribution(mean,std,x)
    powerlaw_cdf,powerlaw_pdf = power_distribution(mean,std,x)
    normal_cdf,normal_pdf = normal_distribution(mean,std,x)

    # distance = stats.wasserstein_distance(normal_cdf,normal_cdf)
    distance_norm = abs(metrics.auc(x,normal_cdf) - metrics.auc(x,normal_cdf))
    # print('normal distance: ', distance_norm)
    # distance = stats.wasserstein_distance(exp_cdf,normal_cdf)
    distance_exp = abs(metrics.auc(x,exp_cdf) - metrics.auc(x,normal_cdf))
    # print('exp distance: ',distance_exp)
    # distance = stats.wasserstein_distance(gamma_cdf,normal_cdf)
    distance_gamma = abs(metrics.auc(x,gamma_cdf) - metrics.auc(x,normal_cdf))
    # print('gamma distance: ',distance_gamma)
    # distance = stats.wasserstein_distance(powerlaw_cdf,normal_cdf)
    distance_powerlaw = abs(metrics.auc(x,powerlaw_cdf) - metrics.auc(x,normal_cdf))
    # print('powerlaw distance: ',distance_powerlaw)
    if distance_exp > distance_powerlaw:
        print('g<p<e',mean/std)
    elif distance_exp >distance_gamma:
        print('g<e<p',mean/std)
    else: 
        print('e<g<p',mean/std) 
        break
    # d = {'dist_names':['normal','exponential','gamma','powerlaw'],'distance':[distance_norm,distance_exp,distance_gamma,distance_powerlaw]}
    # df = pd.DataFrame(data=d)
    # print(mean,std)
    # print(df.sort_values(by=['distance']))
if plot == True:
    fig, (ax1,ax2)  = plt.subplots(1,2)
    distributions = [exp_cdf,gamma_cdf,powerlaw_cdf,normal_cdf]
    labels = ['exponential','gamma','powerlaw','normal']
    for dist,label in zip(distributions,labels):
        ax1.plot(x,dist,label = label)
    #general properties
    ax1.loglog()
    ax1.tick_params(axis='both',labelsize=14)
    ax1.set_xlabel('x', fontsize=15)
    ax1.set_ylabel('CDF',fontsize=15)
    ax1.legend()
    distributions = [exp_pdf,gamma_pdf,powerlaw_pdf,normal_pdf]
    labels = ['exponential','gamma','powerlaw','normal']
    for dist,label in zip(distributions,labels):
        ax2.plot(x,dist,label = label)
    #general properties
    ax2.loglog()
    ax2.tick_params(axis='both',labelsize=14)
    ax2.set_xlabel('x', fontsize=15)
    ax2.set_ylabel('PDF',fontsize=15)
    # ax2.legend()

    plt.tight_layout()
    plt.show()
