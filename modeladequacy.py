import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import data
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.pc = df.pc/np.max(df.pc)
df.Mm = df.Mm/np.max(df.Mm)
y_obs = df.ratio.values

#matrix of independent variables
X = np.column_stack((df.pc.values,df.Mm.values))

#model coefficients
w= [10,10]
w_pc = 10
w_Mm = 10

def model(X,w,y_obs):
    #model/predicted values
    y_pred = (w[0]*X[:,0]+w[1]*X[:,1])**(-1.5)
    #residuals
    residuals = y_obs-y_pred
    #mean squared error
    MSE = np.sum(residuals**2)
    return y_pred, residuals, MSE

def leverage_(X):
    #H=X(X^TX)^-1X^T; yields nxn matrix (12x12)
    H = np.matmul(np.matmul(X,np.linalg.inv(np.matmul(X.transpose(),X))),X.transpose())
    #leverage - diagonal of H
    leverage = np.diagonal(H)
    return leverage

def r_student_(residuals,leverage):
    #standard error
    SE_reg = (MSE*(1-leverage))**(1/2)

    #studentized residuals
    rstudent = residuals/SE_reg
    return rstudent

def cooksdistance(rstudent,leverage,w,MSE):
    #Cook's distance
    D = (rstudent**2/(len(w)*MSE))*(leverage/(1-leverage)**2)
    return D


y_pred, residuals, MSE = model(X,w,y_obs)
leverage =leverage_(X)
rstudent = r_student_(residuals,leverage)

xmin, xmax = np.min(leverage), np.max(leverage)
ymin, ymax = np.min(rstudent), np.max(rstudent)
x = np.linspace(xmin,xmax,100)
y = np.linspace(ymin,ymax,100)
xv, yv = np.meshgrid(x,y)
z = cooksdistance(yv,xv,w,MSE)

#Distribution assumption validiy: QQ plot - observed quantiles vs predicted quantiles

#Residual Analysis - r-student vs predicted
fig,ax = plt.subplots()
ax.scatter(y_pred,rstudent)
ax.hlines(0,0,0.7)
ax.set_xlim(0.01,0.065)
ax.set_ylim(-0.6,0.6)
ax.set_xlabel('Predicted',fontsize=12)
ax.set_ylabel('R-Student',fontsize=12)
ax.tick_params(labelsize=12)
fig.tight_layout()

#Influence Diagnositics - r-student vs leverage (include D=1 and D=0.5 lines)
fig,ax = plt.subplots()
ax.scatter(leverage,rstudent)
ax.contour(x,y,z,levels = 1)
ax.set_xlim(0.01,0.6)
ax.set_ylim(-0.6,0.6)
ax.set_xlabel('leverage',fontsize=12)
ax.set_ylabel('R-Student',fontsize=12)
ax.tick_params(labelsize=12)
fig.tight_layout()

plt.show()