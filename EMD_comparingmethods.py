import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
random.seed(203)
plot_runtime=False
plot_results_comp = False
plot_legend = False
plot_initialfinal = True
plot_EMD_rxnratio = True

res = pd.read_csv("EMD_variations_comparison.csv",header=0,index_col=0)
initial_flow_props = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)

phases = ["initial","final"]
samples = initial_flow_props.index

if plot_runtime:
    fig,ax = plt.subplots()
    bin_min = np.min([np.min(res['built-in_time']),np.min(res['manual-log_time']),np.min(res['manual-linear_time'])])
    bin_max = np.max([np.max(res['built-in_time']),np.max(res['manual-log_time']),np.max(res['manual-linear_time'])])
    bins = 10**np.linspace(np.log10(bin_min),np.log10(bin_max),num=10)
    ax.hist(res['built-in_time'],bins=bins,label="built-in",alpha=0.5)
    ax.hist(res["manual-log_time"],bins=bins,label="manual-log",alpha=0.5)
    ax.hist(res["manual-linear_time"],bins=bins,label="manual-linear",alpha=0.5)
    ax.set_xlabel("Runtime, s",fontsize=15)
    ax.set_ylabel("Frequency",fontsize=15)
    ax.tick_params(labelsize=14)
    ax.semilogx()
    ax.legend()
    fig.tight_layout()
    plt.show()

colors=["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])for j in range(len(res))]

if plot_results_comp:
    # vert line 1: initial, vert line 2: final
    # color: sample
    # line style: comp method
    
    fig,ax= plt.subplots()
    fig2,ax2 = plt.subplots()
    # ax2 = ax.twinx()
    res1 = res.drop(labels=["built-in_time","manual-log_time","manual-linear_time"],axis=1).transpose()
    for ind,sample in enumerate(samples):
        if sample == "port0.1ph3.6":continue
        # elif sample == "port0.1ph3.6":continue
        # elif sample == "AH": continue
        ax2.plot([0,1],[res['manual-log'][sample+"_"+"initial"],res['manual-log'][sample+"_"+"final"]],label=sample,linestyle="-.",color=colors[ind])
        ax.plot([0,1],[res['manual-linear'][sample+"_"+"initial"],res['manual-linear'][sample+"_"+"final"]],label=sample,linestyle="--",color=colors[ind],alpha=0.5)
        ax.plot([0,1],[res['built-in'][sample+"_"+"initial"],res['built-in'][sample+"_"+"final"]],label=sample,linestyle="-",color=colors[ind],alpha=0.5)
    ax.set_xlim(-0.1,1.1)
    ax2.set_xlim(-0.1,1.1)
    ax.semilogy()
    ax2.semilogy()
    ax_ticks = ax.get_xticks()
    ax.set_xticks([0,1],labels=["Initial","Final"],fontsize=14)
    ax2.set_xticks([0,1],labels=["Initial","Final"],fontsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax2.tick_params(axis='y',labelsize=14)
    ax2.set_ylabel("EMD, observed vs lognormal",fontsize=15)
    ax.set_ylabel("EMD, observed vs lognormal",fontsize=15)
    ax.set_title("Evaluated in linear-space",fontsize=16)
    ax2.set_title("Evaluated in log-space",fontsize=16)
    fig.tight_layout()
    fig2.tight_layout()
    plt.show()

if plot_legend:
    # making legend
    custom_lines = []
    for c in colors:
        custom_lines.append(Line2D([0],[0],color=c,lw=4))
    fig,ax = plt.subplots()
    ax.legend(custom_lines,samples,loc="center",ncol=2)
    ax.axis("off")
    fig.tight_layout()    
    custom_lines = []
    for m in ["-","--","-."]:
        custom_lines.append(Line2D([0],[0],color='k',linestyle=m))
    fig2,ax2 = plt.subplots()
    ax2.legend(custom_lines,["built-in","manual-log","manual-linear"],loc="center",ncol=1)
    ax2.axis("off")
    fig2.tight_layout()
    plt.show()