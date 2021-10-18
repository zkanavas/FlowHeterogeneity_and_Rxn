import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

plot_EMD_Pe = True
plot_Vol_hv_Pe = False
plot_Pe_Da = False
plot_Pe = False
plot_Vol_hv = True

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop('menke_ketton',inplace=True)

#response variable
ratio = df.ratio
fig, ax_ = plt.subplots(1,2,figsize=(6,3))

if plot_EMD_Pe == True:
    ax = ax_[0]
    # fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + Vol_hv + EMD:Pe
    intercept = -4.05
    Vol_hv = 9.58
    EMD_Pe = -6.76e-7
    y_pred = np.exp(intercept + Vol_hv*df.Vol_hv + EMD_Pe*df.EMD*df.Pe)

    r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_ylim(0.01,0.065)
    ax.set_xlim(0.01,0.065)
    ax.set_title('y ~ 1 + Vol_hv + EMD:Pe')
    string1 = 'Spearman r: ', str(round(r,2))
    string2 = 'p-value: ',str(round(p,2))
    ax.text(0.015,0.055,''.join(string1))
    ax.text(0.015,0.05,''.join(string2))
    ax.set_ylabel("Reaction Ratio")
    ax.set_xlabel("Predicted Value")
    fig.tight_layout()

if plot_Vol_hv_Pe == True:
    fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + Vol_hv + Vol_hv:Pe
    intercept = -4.46
    Vol_hv = 12.57
    Vol_hv_Pe = 0.0061
    y_pred = np.exp(intercept + Vol_hv*df.Vol_hv + Vol_hv_Pe*df.Vol_hv*df.Pe)

    r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_ylim(0.01,0.065)
    ax.set_xlim(0.01,0.065)
    ax.set_title('y ~ 1 + Vol_hv + Vol_hv:Pe')
    string1 = 'Spearman r: ', str(round(r,2))
    string2 = 'p-value: ',str(round(p,2))
    ax.text(0.015,0.055,''.join(string1))
    ax.text(0.015,0.05,''.join(string2))
    ax.set_ylabel("Reaction Ratio")
    ax.set_xlabel("Predicted Value")
    fig.tight_layout()

if plot_Pe_Da == True:
    fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + Vol_hv + Pe:Da
    intercept = -4.59
    Vol_hv = 13.49
    Pe_Da = 5.28
    y_pred = np.exp(intercept + Vol_hv*df.Vol_hv + Pe_Da*df.Pe*df.Da)

    r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_ylim(0.01,0.065)
    ax.set_xlim(0.01,0.065)
    ax.set_title('y ~ 1 + Vol_hv + Pe:Da')
    string1 = 'Spearman r: ', str(round(r,2))
    string2 = 'p-value: ',str(round(p,2))
    ax.text(0.015,0.055,''.join(string1))
    ax.text(0.015,0.05,''.join(string2))
    ax.set_ylabel("Reaction Ratio")
    ax.set_xlabel("Predicted Value")
    fig.tight_layout()


if plot_Pe == True:
    fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + Vol_hv + Pe
    intercept = -4.50
    Vol_hv = 13.89
    Pe = 1.82e-4
    y_pred = np.exp(intercept + Vol_hv*df.Vol_hv + Pe*df.Pe)

    r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_ylim(0.01,0.065)
    ax.set_xlim(0.01,0.065)
    ax.set_title('y ~ 1 + Vol_hv + Pe')
    string1 = 'Spearman r: ', str(round(r,2))
    string2 = 'p-value: ',str(round(p,2))
    ax.text(0.015,0.055,''.join(string1))
    ax.text(0.015,0.05,''.join(string2))
    ax.set_ylabel("Reaction Ratio")
    ax.set_xlabel("Predicted Value")
    fig.tight_layout()

if plot_Vol_hv == True:
    ax = ax_[1]
    # y ~ 1 + Vol_hv
    # fig,ax = plt.subplots(figsize=(3,3))
    intercept = -4.33
    Vol_hv = 12.39
    y_pred = np.exp(intercept + Vol_hv*df.Vol_hv)
    r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    ax.set_ylim(0.01,0.065)
    ax.set_xlim(0.01,0.065)
    ax.yaxis.set_ticks([])
    ax.set_title('y ~ 1 + Vol_hv')
    string1 = 'Spearman r: ', str(round(r,2))
    string2 = 'p-value: ',str(round(p,2))
    ax.text(0.015,0.055,''.join(string1))
    ax.text(0.015,0.05,''.join(string2))
    # ax.set_ylabel("Reaction Ratio")
    ax.set_xlabel("Predicted Value")
    fig.tight_layout()

plt.show()
