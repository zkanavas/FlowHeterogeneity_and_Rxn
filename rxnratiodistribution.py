import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math
from scipy.special import gamma, gammainc, erf

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop("menke_ketton",axis=0,inplace=True)
ratio = df.ratio.values.tolist()

y,x = np.histogram(ratio,density=True,bins=5) 
x = x[:-1]
mean = np.mean(y)

def gaussian_dist(x,b,a): return (1/(a*(2*3.14)**(1/2)))*np.exp((-1/2)*((x-b)/a)**2)

def binomial_dist(x,b,a): return ((math.factorial(b)/(math.factorial(x)*(math.factorial(b-x))))*a**x*(1-a)**(b-x))

def poisson_dist(x,b,a): return (b**x*np.exp(-b))/(math.factorial(x))

def geometric_dist(x,b): return((1-b)**(x-1)*b)

def neg_binomial_dist(x,b,a): return ((math.factorial(x+b-1)/(math.factorial(x)*(math.factorial(b-1))))*(1-a)**x*a**b)

def exponential_dist(x,b,a): return ((1/a)*np.exp(-(x-b)/a))

def gamma_dist(x,b,a): return ((1/(gamma(b)*a**b))*(x**(b-1))*(math.e**(-x*a)))

def inv_normal_dist(x,b,a): return ((b/(2*math.pi*x**3))**(1/2)*np.exp(-((b*(x-a)**2)/(2*a**2*x))))

def curvefit_plot(dist,x,y,label,color,fig,ax):
    
    w,cov = curve_fit(dist,x,y,maxfev=10000,p0=(1,1))
    print(w)
    # if label == 'geometric': y_pred = dist(x,w)
    # else: 
    y_pred = dist(x,w[0],w[1])
    r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
    line = np.arange(np.min(x),np.max(x),.001)    
    ax.scatter(x,y,c='k')
    lbl = label + " r2: "+str(round(r2,2))
    ax.plot(line,dist(line,w[0],w[1]),label=lbl,lw=3,c=color)
    # ax.text(np.max(x),np.max(y),"r2: "+str(round(r2,2)),ha='right')
    # ax.set_title(label)
    # ax.set_xlabel('rxn ratio')
    # ax.set_ylabel('PDF')
    # fig.tight_layout()
    # plt.show()
    # fig.savefig(label+"dist_fit.png")

colors = ['r','forestgreen','indigo']#,'khaki']#,'mango']#,'forest','cyan','brown']

dists = [gaussian_dist,exponential_dist,inv_normal_dist]

labels = ['normal','exponential','inverse normal']

fig,ax = plt.subplots()
for dist,color,label in zip(dists,colors,labels):
    curvefit_plot(dist,x,y,label,color,fig,ax)
ax.set_xlabel('rxn ratio')
ax.set_ylabel('PDF')
ax.legend()
fig.tight_layout()
plt.show()
# fig,(ax1,ax2) = plt.subplots(1,2,sharey=True)

# w,cov = curve_fit(exponential_dist,x,y)#,p0=(0.07,-.003))
# print(w[0],w[1])
# y_pred = exponential_dist(x,w[0],w[1])
# r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
# print(r2)
# line = np.arange(np.min(x),np.max(x),.001)    
# ax1.scatter(x,y)
# ax1.plot(line,exponential_dist(line,w[0],w[1]),label = "exponential",lw=3,c="orange")
# # eqn = ''.join([str(round(w[0],2)), '*e^(x*', str(round(w[1],2)),')',' r2: ',str(round(r2,2))])
# ax1.text(np.max(x),np.max(y),"r2: "+str(round(r2,2)),ha='right')
# ax1.set_title('Exponential')


# w,cov = curve_fit(gaussian_dist,x,y)#,p0=(0.07,-.003))
# # print(w[0],w[1],np.std(ratio),np.mean(ratio))
# y_pred = gaussian_dist(x,w[0],w[1])
# r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
# print(r2)
# ax2.scatter(x,y)
# ax2.plot(line,gaussian_dist(line,w[0],w[1]),label = "gaussian",lw=3,c="blue")
# ax2.text(np.max(x),np.max(y),"r2: "+str(round(r2,2)),ha='right')
# ax2.set_title('Normal')


# fig.supxlabel('Rxn Ratio')
# fig.supylabel('PDF')
# fig.tight_layout()
# plt.show()
