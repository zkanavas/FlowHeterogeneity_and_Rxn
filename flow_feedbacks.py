import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

initial_flow_props = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
initial_flow_props.drop(['Hinz2019','ketton','Menke2015',"ket0.1ph3.1_sim1","ket0.1ph3.1_sim2","ket0.1ph3.1_sim3","ket0.1ph3.6_sim1","ket0.1ph3.6_sim2","ket0.1ph3.6_sim3","estaillades_sim1","estaillades_sim2"],inplace=True) #don't have final structure for ketton --> from Pereira-Nunes 2016
final_flow_props = pd.read_csv("final_flow_trans_rxn_prop.csv",header=0,index_col=0)

# simulationsamples = ["ket0.1ph3.1_sim1","ket0.1ph3.1_sim2","ket0.1ph3.1_sim3","ket0.1ph3.6_sim1","ket0.1ph3.6_sim2","ket0.1ph3.6_sim3","estaillades_sim1","estaillades_sim2"]
# samplestoremove = ["Menke2015","Hinz2019"]

final_flow_props.drop(["Menke2015","Hinz2019","ket0.1ph3.1_sim1","ket0.1ph3.1_sim2","ket0.1ph3.1_sim3","ket0.1ph3.6_sim1","ket0.1ph3.6_sim2","ket0.1ph3.6_sim3"],inplace=True) #don't have final structure for ketton --> from Pereira-Nunes 2016
samples = initial_flow_props.index

plot_initialfinal=True
plot_flow_rxnratio = False
plot_legend = False

flowmetric = "pc"

if plot_initialfinal:
    size_min = 10
    size_max = 250
    # initial_flow_props[flowmetric]
    # scaled_distances = ((distances.before_after_rescaled - np.min(distances.before_after_rescaled))/np.ptp(distances.before_after_rescaled))*(size_max-size_min)+size_min
    # behavior = ["red" if beh == "uniform" else "blue" for beh in df.behavior]

    fig,ax = plt.subplots()
    # ax.grid(visible=True,zorder=1)
    # for ind,samp in enumerate(samples):
    ind = 0
    for samp in samples:
        # if samp == "BH": print(samp)
        behavior = initial_flow_props.behavior[samp]
       
        if behavior == "wormhole":
            marker = 'D'
        elif behavior == "uniform":
            marker  = 'o'
        elif behavior == "compact":
            marker = 's'
        ax.scatter(initial_flow_props[flowmetric][samp],final_flow_props[flowmetric][samp],marker=marker, c = 'mediumblue', label=samp, alpha = 0.75)
        ind += 1
        print(samp,initial_flow_props[flowmetric][samp],final_flow_props[flowmetric][samp])
    ax.plot([4.725,4.725],[0,15],'k-')
    ax.plot([0,15],[4.725,4.725],'k-')
    lim_factor = np.std(initial_flow_props[flowmetric])/4
    lim_factor_final = np.std(final_flow_props[flowmetric])/4
    ax.set_xlim(1,14)#np.min(initial_flow_props[flowmetric])-lim_factor,np.max(initial_flow_props[flowmetric])+lim_factor)
    ax.set_ylim(1,14)#np.min(final_flow_props[flowmetric])-lim_factor_final,np.max(final_flow_props[flowmetric])+lim_factor_final)
    # ax.plot([0,np.max(initial_flow_props[flowmetric])],[0,np.max(initial_flow_props[flowmetric])],'k-')

    # ax.loglog()
    # ax.semilogy()
    ax.set_xlabel("initial Percolation Threshold",fontsize=15)
    ax.set_ylabel("final Percolation Threshold",fontsize=15)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    # plt.show()
    

if plot_legend:
    scatter_points= []
    for c in ["red","blue","green"]:
        scatter_points.append(Line2D([0],[0],marker='o',color='w',markerfacecolor=c))
    fig,ax = plt.subplots()
    ax.legend(scatter_points,["uniform","wormhole","compact"],loc="center",ncol=1)
    ax.axis("off")
    fig.tight_layout()
    # plt.show()

if plot_flow_rxnratio:
    size_min = 10
    size_max = 250
    before_after = np.abs(final_flow_props[flowmetric] - initial_flow_props[flowmetric])
    scaled_distances = ((before_after - np.min(before_after))/np.ptp(before_after))*(size_max-size_min)+size_min
    # behavior = ["red" if beh == "uniform" else "blue" for beh in df.behavior]

    fig,ax = plt.subplots()
    # ax.grid(visible=True,zorder=1)
    # for ind,samp in enumerate(samples):
    # ind = 0
    for ind,samp in enumerate(samples):
        # if samp == "port0.1ph3.6":continue
        # elif samp == "est0.1ph3.6":continue
        behavior = initial_flow_props.behavior[samp]
        if behavior == "uniform": color = "red" 
        elif behavior == "wormhole": color ="blue" 
        elif behavior == "compact": color = "green"
        ax.scatter(initial_flow_props.ratio[samp],before_after[samp], c = color, s=scaled_distances[ind],label=samp)
        # ind += 1
    # ax.semilogy()
    ax.set_xlabel("Rxn Ratio",fontsize=15)
    ax.set_ylabel("after-before "+flowmetric,fontsize=15)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
plt.show()