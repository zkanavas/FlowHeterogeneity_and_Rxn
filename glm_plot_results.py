import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

plot_Pe_Vol = False
plot_Pe_SA = False
plot_Pe = False
plot_Pe_SSA = False
fig_format = False

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop('menke_ketton',inplace=True)

#response variable
ratio = df.ratio
# fig, ax_ = plt.subplots(1,2,figsize=(6,3))
# fig,ax = plt.subplots()

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
if fig_format == True:
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




#qq
def qq_plot(ax,qq,c):
    ax.scatter(qq.QQ_Assumed,qq.QQ_Response,alpha=0.7,zorder=2,c=c)
    ax.plot([qq.Q1[0],qq.Q3[0]],[qq.Q1[1],qq.Q3[1]],'r-',zorder=0)
    ax.plot([qq.Interp_start[0],qq.Interp_end[0]],[qq.Interp_start[1],qq.Interp_end[1]],'r--',zorder=1)
    ax.tick_params('both',labelsize=13)
    ax.set_xlabel('Normal Quantiles',fontsize=15)
    ax.set_ylabel('Response Quantiles',fontsize=15)

#standard residuals plot
def std_res_plot(ax,df,label,c):
    ax.axhline(y=0,color='lightgray',linestyle='--',zorder=0)
    ax.scatter(df.fitted,df.std_res,alpha=0.7,label=label,zorder=2,c=c)
    ax.plot(sorted(df.fitted),df.lowess_y,zorder=1,c=c)
    ax.tick_params('both',labelsize=13)
    ax.set_xlabel('Fitted Values',fontsize=15)
    ax.set_ylabel('Standardized Residuals',fontsize=15)
    
def cooks_dist_plot(ax,df,label,c,r):
    ax.bar(r,df.cooks_dist,label=label,width=0.3, edgecolor='white',alpha=0.7,color=c)
    ax.axhline(y=0.6667,color='r') #cook's distance threshold
    ax.text(0,0.61,'Threshold',c='r')
    ax.set_ylim(0,0.7)
    # ax.axhline(y=0.5,color='r') #cook's distance threshold
    # ax.text(0,0.45,'Threshold',c='r')
    # ax.set_ylim(0,0.55)
    ax.tick_params('y',labelsize=13)
    # ax.tick_params('x',labelsize=15)
    # ax.set_xticklabels(df.sample_label,fontsize=15)
    loc = [r + 0.3 for r in range(len(df.cooks_dist))]
    labels = ['a','b','c','d','e','f','g','h','i','j']
    ax.set_xticks(loc)
    ax.set_xticklabels(df.sample_label,fontsize=15)
    ax.set_xlabel('Sample Label',fontsize=15)
    ax.set_ylabel("Cook's Distance",fontsize=15)

pc_PeVol = pd.read_csv('pc_PeVol.csv',header=0)
pc_Pe = pd.read_csv('pc_Pe.csv',header=0)
pc_SA = pd.read_csv('pc_PeSA.csv',header=0)
pc_SSA = pd.read_csv('pc_PeSSA.csv',header=0)
EMD = pd.read_csv('EMD.csv',header=0)

#create diagnostic plots

qq = pd.read_csv('qq_log_trans.csv',header=0)

Pe_Vol_label = r'$\mu = \beta_0 + \beta_1 p_c + \beta_2 Pe:Vol_{HVR}$ AIC: 4.29'
Pe_SA_label = r'$\mu = \beta_0 + \beta_1 p_c + \beta_2 Pe:SA_{HVR}$ AIC: 4.33'
Pe_label = r'$\mu = \beta_0 + \beta_1 p_c + \beta_2 Pe$ AIC: 4.46'
# Pe_SSA_label = r'$\mu = \beta_0 + \beta_1 p_c + \beta_2 Pe:SSA_{HVR}$ AIC: '

#x-position of bars for cook's distance plot
barWidth = 0.3
r1 = np.arange(len(pc_PeVol.cooks_dist))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12, 4))
for data,label,c,r in zip([pc_PeVol,pc_SA,pc_Pe],[Pe_Vol_label,Pe_SA_label,Pe_label],['b','red','green'],[r1,r2,r3]):
    qq_plot(ax1,qq,c)
    std_res_plot(ax2,data,label,c)
    cooks_dist_plot(ax3,data,label,c,r)

# for data,label,c in zip([EMD],[r'$\mu = \beta_0 + \beta_1 EMD$ AIC: 13.00'],['purple']):
#     qq_plot(ax1,qq,c)
#     std_res_plot(ax2,data,label,c)
#     cooks_dist_plot(ax3,data,label,c)

ax2.set_ylim(-1,1)
# ax2.legend(loc='lower center',bbox_to_anchor=(0.5,1.1))
# ax3.legend()
fig.tight_layout()
plt.show()
