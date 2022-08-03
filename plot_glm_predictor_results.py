import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from labellines import labelLine, labelLines
from matplotlib.lines import Line2D
from matplotlib import colors

plot_predvsobs = False
plot_heatmap = True
plot_scatter = False
do_legend = False
#import data
# df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df = pd.read_csv('flow_transport_finalrxnrate.csv',header=0,index_col=0)
simulationsamples = ["ket0.1ph3.1_sim1","ket0.1ph3.1_sim2","ket0.1ph3.1_sim3","ket0.1ph3.6_sim1","ket0.1ph3.6_sim2","ket0.1ph3.6_sim3","estaillades_sim1","estaillades_sim2"]
samplestoremove = ["est0.1ph3.6","port0.1ph3.1","port0.1ph3.6"]
df.drop(samplestoremove +simulationsamples,axis=0,inplace=True) #,9,14

#models
def single_predictor(x0,x1,x2,pc,Pe):
    y = np.exp(x0 + x1*pc+ x2*Pe)
    return y

# min_pc,max_pc = df.pc.min(),df.pc.max()
# min_PeVol,max_Pe = min(df.Pe*df.Vol_hv),max(df.Pe*df.Vol_hv)

def compute_prediction(min_pc,max_pc,min_Pe,max_Pe):
    # beta0 = -2.4535
    # beta1 = -0.21094
    # beta2 = 0.00033157
    # beta0=-2.2163
    # beta1=-0.35445
    # beta2=0.00083158
    beta0 = -1.9535
    beta1 = -0.4088
    beta2 = 8.7325e-4
    color_list = []
    for Pe in np.linspace(min_Pe,max_Pe,100):
        for pc in np.linspace(min_pc,max_pc,100):
            # color_list.append(single_predictor(-2.38,-0.30,5.42e-4,pc,Pe)) #2-model
            color_list.append(single_predictor(beta0,beta1,beta2,pc,Pe)) #3-model #
    return color_list

if plot_scatter == True:
    
    min_pc,max_pc = 1,10
    min_Pe,max_Pe = 100,2100
    color_list = compute_prediction(min_pc,max_pc,min_Pe,max_Pe)

    plot_data = np.reshape(color_list,(100,100))

    levels = np.geomspace(np.min(color_list),np.max(color_list),6)

    # fig,ax = plt.subplots()

    colors_ = ['red' if x =='uniform' else 'blue' for x in df.behavior]

    start = 100
    end = 1000
    width = end - start
    res = (df.ratio - df.ratio.min())/(df.ratio.max() - df.ratio.min()) * width + start


    cntr = plt.contour(np.linspace(min_pc,max_pc,100),np.linspace(min_Pe,max_Pe,100),plot_data,levels=levels,colors='k')
    plt.close()

    fig,ax = plt.subplots()
    sctr = ax.scatter(df.pc,df.Pe,c=colors_,s=res)

    for i,level in enumerate(levels):
        if i == 0 or i == 5:
            continue
        ax.plot(cntr.allsegs[i][0][:,0],cntr.allsegs[i][0][:,1],label=str(round(level,2)),c='k')


    labelLines(plt.gca().get_lines())

    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel('Percolation Threshold', fontsize=15)
    ax.set_ylabel('Peclet Number',fontsize=15)

    if do_legend:
        # produce a legend with the unique colors from the scatter
        legend_elements = [Line2D([0], [0], marker='o',color='w', label='Uniform',
                                markerfacecolor='red', markersize=10),
                        Line2D([0], [0], marker='o',color='w', label='Wormhole',
                                markerfacecolor='blue', markersize=10)]
        # legend1 = ax.legend(handles=legend_elements,loc="lower left",bbox_to_anchor=(0.97, 0.5), title="Behavior",labelspacing=1,frameon=False)
        # ax.add_artist(legend1)
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = sctr.legend_elements(prop="sizes")
        labels = [round(df.ratio.min(),2),round(df.ratio.mean(),2),round(df.ratio.max(),2)]
        legend2 = ax.legend([handles[0],handles[3],handles[-1]], labels, loc="lower left",bbox_to_anchor=(1.01, 0), title=r"$r_e/r_b$",labelspacing=2,frameon=False)

    ax.set_ylim(50,2100)
    ax.set_xlim(1,10)
    fig.tight_layout()

# plt.show()

if plot_heatmap == True:
    min_pc,max_pc = 1,10
    min_Pe,max_Pe = 2100,100
    color_list = compute_prediction(min_pc,max_pc,min_Pe,max_Pe)

    plot_data = np.reshape(color_list,(100,100))

    divnorm = colors.TwoSlopeNorm(vmin=0,vmax=0.5,vcenter=0.075)
    # fig,ax = plt.subplots()
    ax2 = sns.heatmap(plot_data,xticklabels=10,yticklabels=10,cmap = "seismic",norm=divnorm,cbar_kws = {'label':r'mean($r_e/r_b$)'})

    # use matplotlib.colorbar.Colorbar object
    cbar = ax2.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_ylabel(ylabel=r'mean($r_e/r_b$)',fontsize=15)

    xticklabels = [int(x) for x in np.linspace(min_pc,max_pc,10)]
    yticklabels = [int(x) for x in np.linspace(min_Pe,max_Pe,10)]

    ax2.set_xticklabels(xticklabels)
    ax2.set_yticklabels(yticklabels,rotation = 'horizontal')
    ax2.tick_params(axis='both',labelsize=14)
    ax2.set_xlabel('Percolation Threshold', fontsize=15)
    ax2.set_ylabel('Peclet Number',fontsize=15)
    plt.tight_layout()

plt.show()