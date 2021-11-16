import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df.drop(3,axis=0,inplace=True)

colors = ['red' if x =='uniform' else 'blue' for x in df.behavior]

start = 100
end = 1000
width = end - start
res = (df.ratio - df.ratio.min())/(df.ratio.max() - df.ratio.min()) * width + start

fig,ax = plt.subplots()
sctr = ax.scatter(df.pc,df.Pe*df.Vol_hv,c=colors,s=res)

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

plt.show()


