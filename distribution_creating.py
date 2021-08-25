#distributions
# from scipy.stats import wasserstein_distance#,powerlaw,gamma,norm,exp
import scipy.stats as stats
from scipy.special import gamma, gammainc, erf
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.metrics as metrics


plot = True
#exponential, gamma, powerlaw
rng = np.random.default_rng(12345)
def exponential_distribution(mean,std,x):
    lambda_ = 1/mean
    exp_cdf = 1 - math.e**(-lambda_*x)
    exp_cdf_func = stats.expon.cdf(x,scale = mean)
    exp_pdf = stats.expon.pdf(x,scale = mean)
    exp_pdf = (lambda_) * math.e**(-(x)*lambda_)
    cumlsum = np.cumsum(exp_pdf)
    exp_cdf = cumlsum/cumlsum[-1]
    if all(np.logical_or(exp_cdf - 1e-5 <= exp_cdf_func, exp_cdf + 1e-5 >= exp_cdf_func)):
        return exp_cdf,exp_pdf
    else: return print('error in exponential distribution')

def gamma_distribution(mean,std,x):
    theta = std**2/mean
    k = mean/theta
    gamma_cdf = gammainc(k,(x/theta))/gamma(k)
    gamma_pdf = (1/(gamma(k)*theta**k))*(x**(k-1))*(math.e**(-x*theta))
    cumlsum = np.cumsum(gamma_pdf)
    gamma_cdf = cumlsum/cumlsum[-1]
    # gamma_cdf_func = stats.gamma.cdf(x,k,scale=theta)
    # gamma_pdf = stats.gamma.pdf(x,k,scale=theta)
    # if all(np.logical_or(gamma_cdf - 1e-5 <= gamma_cdf_func, gamma_cdf + 1e-5 >= gamma_cdf_func)):
    return gamma_cdf, gamma_pdf
    # else: return print('error in gamma distribution')

def powerlaw_distribution(mean,std,x):
    xm = np.min(x)
    #well-defined mean: alpha > 2, finite variance: mean > 3;
    alpha = (3-((2*xm*mean)/std**2))/(1-((xm*mean)/std**2))
    if alpha < 3:
        return print('error! alpha is less than 3')
    powerlaw_cdf = stats.powerlaw.cdf(x,alpha,scale=xm)
    powerlaw_pdf = (alpha)*(x)**(-alpha)
    cumulativesum = np.cumsum(powerlaw_pdf)
    powerlaw_cdf = cumulativesum/cumulativesum[-1]
    return powerlaw_cdf, powerlaw_pdf

def normal_distribution(mean,std,x):
    normal_cdf = (1/2)*(1+erf((x-mean)/(std*(2)**(1/2))))
    normal_cdf_func = stats.norm.cdf(x,loc=mean,scale=std)
    normal_pdf = stats.norm.pdf(x,loc=mean,scale=std)
    if all(np.logical_or(normal_cdf - 1e-5 <= normal_cdf_func, normal_cdf + 1e-5 >= normal_cdf_func)):
        return normal_cdf_func, normal_pdf
    else: return print('error in normal distribution')   

x = np.linspace(1e-3,1e3,100000)
mean = 1.5
std = 1.5

# fig, ax = plt.subplots()

exp_cdf,exp_pdf = exponential_distribution(mean,std,x)
powerlaw_cdf, powerlaw_pdf = powerlaw_distribution(mean,std,x)
normal_cdf,normal_pdf = normal_distribution(mean,std,x)
gamma_cdf,gamma_pdf = gamma_distribution(mean,std,x)

distance_norm = abs(metrics.auc(x,normal_cdf) - metrics.auc(x,normal_cdf))
print('normal distance: ', distance_norm)

distance_exp = abs(metrics.auc(x,exp_cdf) - metrics.auc(x,normal_cdf))
print('exp distance: ',distance_exp)

distance_gamma = abs(metrics.auc(x,gamma_cdf) - metrics.auc(x,normal_cdf))
print('gamma distance: ',distance_gamma)

distance_powerlaw = abs(metrics.auc(x,powerlaw_cdf) - metrics.auc(x,normal_cdf))
print('powerlaw distance: ',distance_powerlaw)

if plot == True:
    
    cdf_distributions = [exp_cdf,gamma_cdf,powerlaw_cdf]
    pdf_distributions = [exp_pdf,gamma_pdf,powerlaw_pdf]
    labels = ['exponential','gamma','powerlaw']
    colors = ['green','orange','purple']
    fig, ax1 = plt.subplots()
    for ind, dist in enumerate(cdf_distributions):
        fig, (ax1,ax2)  = plt.subplots(1,2, figsize = [12.8, 4.8])
        ax1.plot(x,dist,label = labels[ind],color=colors[ind])
        ax1.plot(x,normal_cdf,label='normal',color='blue')
        ax1.fill_between(x,normal_cdf,dist,color='yellow')
        ax2.plot(x,pdf_distributions[ind],label = labels[ind],color=colors[ind])
        ax2.plot(x,normal_pdf,label='normal',color='blue')

        #general properties
        ax1.loglog()
        ax1.tick_params(axis='both',labelsize=14)
        ax1.set_xlabel('x', fontsize=15)
        ax1.set_ylabel('CDF',fontsize=15)
        ax1.legend()
        ax2.loglog()
        ax2.tick_params(axis='both',labelsize=14)
        ax2.set_xlabel('x', fontsize=15)
        ax2.set_ylabel('PDF',fontsize=15)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(labels[ind]+"_normal_comp.png")
        plt.close()
