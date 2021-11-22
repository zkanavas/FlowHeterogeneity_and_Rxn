import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from SALib.sample import saltelli
from SALib.analyze import sobol

def model(pc,Pe,beta_0,beta_1,beta_2):
    return np.exp(beta_0 + beta_1*pc + beta_2*Pe)

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop('menke_ketton',inplace=True)

problem = { 'num_vars':3,
            'names':['beta_0','beta_1','beta_2'],
            'bounds':[[-3.2726, -2.1156],[-0.3460, -0.1408],[0.0009, 0.0271]]}

# problem = { 'num_vars':3,
#             'names':['beta_0','beta_1','beta_2'],
#             'bounds':[[-2.9847, -1.7798],[-0.4300, -0.1707],[0.0000, 0.0011]]}

# sample
param_values = saltelli.sample(problem, 2**6)

# evaluate
# pc = np.linspace(np.min(df.pc),np.max(df.pc),100)
# Pe = np.mean(df.Pe)
pc = np.mean(df.pc)
Pe = np.linspace(np.min(df.Pe*df.Vol_hv),np.max(df.Pe*df.Vol_hv),100)

print(pc,np.mean(df.Pe*df.Vol_hv),np.mean(df.Pe))

y = np.array([model(pc,Pe, *params) for params in param_values])

#analyze
sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

S1s = np.array([s['S1'] for s in sobol_indices])

fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = fig.add_gridspec(3, 2)

ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, 1])

for i, ax in enumerate([ax1, ax2, ax3]):
    ax.plot(Pe, S1s[:, i],
            label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]),
            color='black')
    ax.set_xlabel("$Pe:Vol_{HVR}$")
    ax.set_ylabel("First-order Sobol index")

    ax.set_ylim(0, 1.04)

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.legend(loc='upper right')

ax0.plot(Pe, np.mean(y, axis=0), label="Mean", color='black')

# in percent
prediction_interval = 95

ax0.fill_between(Pe,
                 np.percentile(y, 50 - prediction_interval/2., axis=0),
                 np.percentile(y, 50 + prediction_interval/2., axis=0),
                 alpha=0.5, color='black',
                 label=f"{prediction_interval} % prediction interval")

ax0.set_ylim(0, 0.175)
ax0.set_xlabel("$Pe:Vol_{HVR}$")
ax0.set_ylabel("rxn ratio")
ax0.legend(title=r"$y=\beta_0+\beta_1\cdot p_c+\beta_2\cdot Pe:Vol_{HVR}$",
           loc='upper center')._legend_box.align = "left"
plt.tight_layout()
# plt.show()