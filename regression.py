from numpy.ma.extras import flatnotmasked_contiguous
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

poly = True
exp = True
powerlaw = True
method = 'lm'

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df.drop(3,axis=0,inplace=True)

heterogeneities = ["EMD"]
for heterogeneity in heterogeneities:
    x = df[heterogeneity].values.tolist()
    y = df['ratio'].values.tolist()
    fig, ax = plt.subplots()
    # if heterogeneity != "pc": poly =True
    if poly == True:        
        # polyreg = np.poly1d(np.polyfit(x,y,3))
        def func(b,a,t,deg): return a+t*(b)**(deg)
        w,cov = curve_fit(func,x,y,p0=(0.05,-0.01,0.18),method=method,maxfev=10000)
        polyline = np.arange(np.min(x),np.max(x),.1)
        # ax.scatter(x,y)
        # ax.plot(polyline,func(polyline,0.05,-0.01,0.18),label="edu_guess")
        ax.plot(polyline,func(polyline,w[0],w[1],w[2]),label = "polynomial",lw=3,c="blue")
        # w = [0.05,-0.01,0.18]
        y_pred = w[0]+w[1]*np.power(x,w[2])
        mean = np.mean(y)
        r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
        eqn = ''.join([str(round(w[0],2)), '+', str(round(w[1],2)),'x^',str(round(w[2],2)),' r2: ',str(round(r2,2))])
        ax.text(np.max(x),0.05,eqn,c="blue",fontweight='bold',ha='right')

    if exp == True:
        # expreg = np.poly1d(np.polyfit(x,np.log(y),1))
        def func(b,a,t): return a*np.exp(b*t)
        w,cov = curve_fit(func,x,y,p0=(0.07,-.003),method=method)
        expline = np.arange(np.min(x),np.max(x),.1)    
        # ax.scatter(x,y)
        # ax.plot(expline,func(expline,0.07,-0.003),label="edu_guess")
        ax.plot(expline,func(expline,w[0],w[1]),label = "exponential",lw=3,c="orange")
        # w = [0.07,-0.003]
        y_pred = w[0]*np.exp(np.multiply(x,w[1]))
        mean = np.mean(y)
        r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
        eqn = ''.join([str(round(w[0],2)), '*e^(x*', str(round(w[1],2)),')',' r2: ',str(round(r2,2))])
        ax.text(np.max(x),0.045,eqn,c="orange",fontweight='bold',ha='right')

    if powerlaw == True:
        # expreg = np.poly1d(np.polyfit(x,np.log(y),1))
        def func(b,a,t): return a*(b)**t
        w,cov = curve_fit(func,x,y,p0=(0.07,-0.1),method=method)
        powerline = np.arange(np.min(x),np.max(x),.1)    
        ax.scatter(x,y,c="k")
        # ax.plot(powerline,func(powerline,0.03,-0.15),label="edu_guess")
        ax.plot(powerline,func(powerline,w[0],w[1]),label = "powerlaw",lw=3,c="green")
        y_pred = w[0]*np.power(x,w[1])
        mean = np.mean(y)
        r2 = 1 - (np.sum((y_pred-y)**2))/(np.sum((y-mean)**2))
        eqn = ''.join([str(round(w[0],2)), '*x^', str(round(w[1],2)),' r2: ',str(round(r2,2))])
        ax.text(np.max(x),0.04,eqn,c="green",fontweight='bold',ha='right')
        # print(cov)
    ax.semilogx()
    ax.tick_params("both",labelsize=14)
    ax.set_xlabel(heterogeneity,fontsize=15)
    ax.set_ylabel("Rxn Ratio",fontsize=15)
    plt.legend()
    plt.tight_layout()
plt.show()