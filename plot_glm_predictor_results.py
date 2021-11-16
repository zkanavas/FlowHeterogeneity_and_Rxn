import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from labellines import labelLine, labelLines
from matplotlib.lines import Line2D

plot_predvsobs = False
plot_heatmap = True
plot_scatter = True

#import data
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df.drop(3,axis=0,inplace=True)

#models

def predictor(x0,x1,x2,df,extra):
    y = np.exp(x0 + x1*df.pc+ x2*df.Pe*extra)
    return y

def single_predictor(x0,x1,x2,pc,Pe):
    y = np.exp(x0 + x1*pc+ x2*Pe)
    return y

# min_pc,max_pc = df.pc.min(),df.pc.max()
# min_PeVol,max_Pe = min(df.Pe*df.Vol_hv),max(df.Pe*df.Vol_hv)

def compute_prediction(min_pc,max_pc,min_Pe,max_Pe):
    color_list = []
    for Pe in np.linspace(min_Pe,max_Pe):
        for pc in np.linspace(min_pc,max_pc):
            # color_list.append(single_predictor(-2.38,-0.30,5.42e-4,pc,Pe)) #2-model
            color_list.append(single_predictor(-2.69,-0.24,0.014,pc,Pe)) #3-model
    return color_list

if plot_predvsobs == True:
    #create figures
    fig, ax = plt.subplots()

    # ax1[0].scatter(df.ratio,predictor(-2.69,-0.24,0.014,df,df.Vol_hv))
    # ax1[0].title('y ~ exp(1+ pc + Pe*Vol)')

    # ax1[1].scatter()
    # ax1[2].title('y ~ exp(1+ pc + Pe*SA)')

    ax.scatter(df.ratio,predictor(-2.38,-0.30,5.42e-4,df,np.ones(len(df.Pe))))
    # ax2[0].set_title('y ~ exp(1+ pc + Pe)')

    # ax2[1].scatter()
    # ax2[1].title('y ~ exp(1+ pc + Pe*SSA)')
    # plt.show()
    fig.tight_layout()

if plot_scatter == True:
    
    min_pc,max_pc = 1,10
    min_Pe,max_Pe = 1,50
    color_list = compute_prediction(min_pc,max_pc,min_Pe,max_Pe)

    plot_data = np.reshape(color_list,(50,50))

    levels = np.geomspace(np.min(color_list),np.max(color_list),6)

    # fig,ax = plt.subplots()

    colors = ['red' if x =='uniform' else 'blue' for x in df.behavior]

    start = 100
    end = 1000
    width = end - start
    res = (df.ratio - df.ratio.min())/(df.ratio.max() - df.ratio.min()) * width + start


    cntr = plt.contour(np.linspace(min_pc,max_pc),np.linspace(min_Pe,max_Pe),plot_data,levels=levels,colors='k')
    plt.close()

    fig,ax = plt.subplots()
    sctr = ax.scatter(df.pc,df.Pe*df.Vol_hv,c=colors,s=res)

    for i,level in enumerate(levels):
        if i == 0 or i == 5:
            continue
        ax.plot(cntr.allsegs[i][0][:,0],cntr.allsegs[i][0][:,1],label=str(round(level,2)),c='k')


    labelLines(plt.gca().get_lines())

    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel('pc', fontsize=15)
    ax.set_ylabel('Pe*Vol',fontsize=15)

    # produce a legend with the unique colors from the scatter
    legend_elements = [Line2D([0], [0], marker='o',color='w', label='Uniform',
                            markerfacecolor='red', markersize=10),
                    Line2D([0], [0], marker='o',color='w', label='Wormhole',
                            markerfacecolor='blue', markersize=10)]
    legend1 = ax.legend(handles=legend_elements,loc="lower left",bbox_to_anchor=(1.01, 0.5), title="Behavior",labelspacing=1,frameon=False)
    ax.add_artist(legend1)
    # produce a legend with a cross section of sizes from the scatter
    handles, labels = sctr.legend_elements(prop="sizes")
    labels = [df.ratio.min(),df.ratio.max()]

    legend2 = ax.legend([handles[0],handles[-1]], labels, loc="lower left",bbox_to_anchor=(1.01, 0), title="Rxn Ratio",labelspacing=2,frameon=False)
    fig.tight_layout()


if plot_heatmap == True:
    min_pc,max_pc = 1,10
    min_Pe,max_Pe = 50,1
    color_list = compute_prediction(min_pc,max_pc,min_Pe,max_Pe)

    plot_data = np.reshape(color_list,(50,50))

    fig,ax = plt.subplots()
    sns.heatmap(plot_data,xticklabels=5,yticklabels=5,cbar_kws = {'label':'Rxn Ratio'},ax=ax)

    xticklabels = [int(x) for x in np.linspace(min_pc,max_pc,10)]
    yticklabels = [int(x) for x in np.linspace(min_Pe,max_Pe,10)]

    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels,rotation = 'horizontal')

    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel('pc', fontsize=15)
    ax.set_ylabel('Pe*Vol',fontsize=15)

    fig.tight_layout()

plt.show()