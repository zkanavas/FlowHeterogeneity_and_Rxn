import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def moving_average(a, n=2) : #menkelikesmovingaverageof2
    ret = np.nancumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def check_convergence(roc_array,threshold=3e-3):
    for ind,r in enumerate(roc_array):
        if all(roc_array[ind:] < threshold):
            # print("converged")
            return ind
        else:
            continue

plot_figuredata = False
plot_rxn_rate = False
plot_rateofchange = True

allsampleinfo = pd.read_csv("flow_transport_rxn_properties.csv",header=0,index_col=0)
# intermstructureinfo = pd.read_csv("Publication_intermediate_structures.csv", header=0,index_col=1)
SSAporosityinfo = pd.read_csv("sample_SSA_porosity_intime.csv",header=0,index_col=0) #time is in minutes!!

figuredata = pd.read_csv(r"C:\Users\zkana\Downloads\menke2017.csv",header=[0,1])
sampnames = ['est3.1','est3.6','ket3.1','ket3.6','port3.1','port3.6']
# figuredata = pd.read_csv(r"C:\Users\zkana\Downloads\alkhulaifi2018.csv",header=[0,1])
# sampnames = ['HetBHigh','HetAHigh','HetALow','HetBLow']
# figuredata = pd.read_csv(r"PereiraNunes2016.csv",header=[0,1])
# sampnames = ['beadpack','estaillades']

colors = ['blue','red','green','purple','orange','navy','black','gray','brown','pink']
ymax=1.2e-4
fig,ax = plt.subplots()
if plot_figuredata:
    for ind,sampname in enumerate(sampnames):
        # if "ket3.6" in sampname: continue
        averagedreff = moving_average(figuredata[sampname].Y.dropna())
        averagedtime = moving_average(figuredata[sampname].X.dropna())
        changeiny = abs(np.diff(averagedreff))/averagedreff[0]
        changeinx = abs(np.diff(averagedtime))
        rateofchange = changeiny/changeinx
        index = check_convergence(rateofchange)
        if index == None: 
            print("no convergence for sample ", sampname)
        else:
            print(sampname, "reff: ", np.mean(figuredata[sampname].Y.dropna()[index:]), " at ", figuredata[sampname].X.dropna()[index])
            ax.plot(figuredata[sampname].X.dropna(),figuredata[sampname].Y.dropna(),'-',color=colors[ind],label=sampname)
            ax.vlines(figuredata[sampname].X.dropna()[index],color=colors[ind],ymin=0,ymax=ymax)
        print(rateofchange)

count=0
for ind,sample in enumerate(SSAporosityinfo.index):    
    # if "Menke2017" not in SSAporosityinfo.Publication[sample]:continue
    # if "ket0.1ph3.6" in sample: 
    #     count+=1
    #     continue
    if "_sim" not in sample: continue
    reff = [float(ele) for ele in SSAporosityinfo.reff[sample][1:-1].split(",")]
    timesteps = [float(ele) for ele in SSAporosityinfo.Timestep[sample][1:-1].split(",")][1:] #in seconds
    averagedreff = moving_average(reff)
    averagedtime = moving_average(timesteps)
    changeiny = abs(np.diff(averagedreff))/averagedreff[0]
    changeinx = abs(np.diff(averagedtime))
    rateofchange = changeiny/changeinx
    index = check_convergence(rateofchange)
    if index == None: 
        print("no convergence for sample ", sample)
    else:
        # print(sample, "reff: ", np.mean(reff[index:]), " at ", timesteps[index])
        if plot_rxn_rate:
            ax.plot(timesteps,reff,'-.',color=colors[count],label=sample)
            # ax.vlines(timesteps[index],color=colors[count],ymin=0,ymax=ymax)
        elif plot_rateofchange:
            ax.plot(averagedtime[1:],rateofchange,'-.',label=sample)
            
        count +=1


# ax.set_ylim(0,ymax)
ax.set_xlabel("Time [min]")
if plot_rxn_rate:
    ax.set_ylabel("Effective Reaction Rate (reff) [mol/m2s]")
elif plot_rateofchange:
    ax.hlines(3e-3,color='black',xmin=0,xmax=300)
    ax.set_ylabel("Rate of Change")
    ax.set_xlim(0,75)
ax.legend()
fig.tight_layout()
plt.show()