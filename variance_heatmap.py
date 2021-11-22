import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

complex = True
matrix_resolution = 100
CI_resolution = 10000
# rng = np.random.default_rng(155)

def model(pc,Pe,beta_0,beta_1,beta_2):
    return np.exp(beta_0 + beta_1*pc + beta_2*Pe)
if complex == True:
    #CIs - for pc + Pe;Vol_hv
    beta0_CI = np.linspace(-3.2726, -2.1156,CI_resolution)
    beta1_CI = np.linspace(-0.3460, -0.1408,CI_resolution)
    beta2_CI = np.linspace(0.0009, 0.0271,CI_resolution)
    min_Pe,max_Pe = 50,1
else:
    #CIs - for pc + Pe
    beta0_CI = np.linspace(-2.9847, -1.7798,CI_resolution)
    beta1_CI = np.linspace(-0.4300, -0.1707,CI_resolution)
    beta2_CI = np.linspace(0.0000, 0.0011,CI_resolution)
    min_Pe,max_Pe = 2100,100

# np.random.shuffle(beta0_CI)
# np.random.shuffle(beta1_CI)
# np.random.shuffle(beta2_CI)


min_pc,max_pc = 1,10

var = []
for Pe in np.linspace(min_Pe,max_Pe,matrix_resolution):
    for pc in np.linspace(min_pc,max_pc,matrix_resolution):
        # color_list.append(model(pc,Pe,-2.38,-0.30,5.42e-4)) #2-model
        np.random.shuffle(beta0_CI)
        np.random.shuffle(beta1_CI)
        np.random.shuffle(beta2_CI)
        var.append(np.var(model(pc,Pe,beta0_CI,beta1_CI,beta2_CI))) #3-model

plot_data = np.reshape(var,(matrix_resolution,matrix_resolution))

fig,ax = plt.subplots()
sns.heatmap(plot_data,xticklabels=int(matrix_resolution/10),yticklabels=int(matrix_resolution/10),cbar_kws = {'label':'Rxn Ratio Variance'},ax=ax,norm=LogNorm(vmin=1e-5,vmax = 0.1))

xticklabels = [int(x) for x in np.linspace(min_pc,max_pc,10)]
yticklabels = [int(x) for x in np.linspace(min_Pe,max_Pe,10)]

ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels,rotation = 'horizontal')

ax.tick_params(axis='both',labelsize=14)
ax.set_xlabel('$p_c$', fontsize=15)
if complex == True:
    ax.set_ylabel('$Pe:Vol_{HVR}$',fontsize=15)
else:
    ax.set_ylabel('$Pe$',fontsize=15)

fig.tight_layout()

plt.show()