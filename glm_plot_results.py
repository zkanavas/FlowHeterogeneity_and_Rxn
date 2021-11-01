import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

plot_Pe_Vol = True
plot_Pe_SA = False
plot_Pe = True
plot_Pe_SSA = False

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop('menke_ketton',inplace=True)

#response variable
ratio = df.ratio
# fig, ax_ = plt.subplots(1,2,figsize=(6,3))
fig,ax = plt.subplots()

if plot_Pe_Vol == True:
    # ax = ax_[0]
    # fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + pc + Pe:Vol_hv
    intercept = -2.6941
    pc = -0.24341
    Pe_Vol = 0.013992
    y_pred = np.exp(intercept + pc*df.pc + Pe_Vol*df.Pe*df.Vol_hv)

    # r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio,label='y ~ exp(1 + pc + Pe:Vol_hv)',alpha=0.6)
    # ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    # ax.set_ylim(0.01,0.065)
    # ax.set_xlim(0.005,0.065)
    # ax.set_title('y ~ exp(1 + pc + Pe:Vol_hv)')
    # string1 = 'Spearman r: ', str(round(r,2))
    # string2 = 'p-value: ',str(round(p,2))
    # ax.text(0.015,0.055,''.join(string1))
    # ax.text(0.015,0.05,''.join(string2))
    # ax.set_ylabel("Reaction Ratio")
    # ax.set_xlabel("Predicted Value")
    # fig.tight_layout()

if plot_Pe_SA == True:
    # fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + pc + Pe:SA_hv
    intercept = -2.7619
    pc = -0.22712
    Pe_SA = 8.3383e-3
    y_pred = np.exp(intercept + pc*df.pc + Pe_SA*df.Pe*df.SA_hv)

    # r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio,label='y ~ exp(1 + pc + Pe:SA_hv)',alpha=0.6)
    # ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    # ax.set_ylim(0.01,0.065)
    # ax.set_xlim(0.005,0.065)
    # ax.set_title('y ~ exp(1 + pc + Pe:SA_hv)')
    # string1 = 'Spearman r: ', str(round(r,2))
    # string2 = 'p-value: ',str(round(p,2))
    # ax.text(0.015,0.055,''.join(string1))
    # ax.text(0.015,0.05,''.join(string2))
    # ax.set_ylabel("Reaction Ratio")
    # ax.set_xlabel("Predicted Value")
    # fig.tight_layout()

if plot_Pe == True:
    # fig, ax = plt.subplots(figsize=(3,3))
    # y ~ 1 + pc + Pe
    intercept = -2.3823
    pc = -0.30036
    Pe = 5.4184e-4
    y_pred = np.exp(intercept + pc*df.pc + Pe*df.Pe)

    # r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio,label='y ~ exp(1 + pc + Pe)',alpha=0.6)
    # ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    # ax.set_ylim(0.01,0.065)
    # ax.set_xlim(0.005,0.065)
    # ax.set_title('y ~ exp(1 + pc + Pe)')
    # string1 = 'Spearman r: ', str(round(r,2))
    # string2 = 'p-value: ',str(round(p,2))
    # ax.text(0.015,0.055,''.join(string1))
    # ax.text(0.015,0.05,''.join(string2))
    # ax.set_ylabel("Reaction Ratio")
    # ax.set_xlabel("Predicted Value")
    # fig.tight_layout()

if plot_Pe_SSA == True:
    # ax = ax_[1]
    # y ~ 1 + pc + Pe:SSA
    # fig,ax = plt.subplots(figsize=(3,3))
    intercept = -2.4465
    pc = -0.28332
    Pe_SSA = 3.2145e-4
    y_pred = np.exp(intercept + pc*df.pc + Pe_SSA*df.Pe*df.SSA)
    # r,p = stats.spearmanr(df.ratio,y_pred)

    ax.scatter(y_pred,ratio,label='y ~ exp(1 + pc + Pe:SSA)',alpha=0.6)
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.set_ylim(0.01,0.065)
ax.set_xlim(0.005,0.065)
# ax.yaxis.set_ticks([])
# ax.set_title('y ~ exp(1 + pc + Pe:SSA)')
# string1 = 'Spearman r: ', str(round(r,2))
# string2 = 'p-value: ',str(round(p,2))
# ax.text(0.015,0.055,''.join(string1))
# ax.text(0.015,0.05,''.join(string2))
ax.set_ylabel("Reaction Ratio",fontsize=15)
ax.set_xlabel("Predicted Value",fontsize=15)
ax.legend()
fig.tight_layout()

plt.show()
