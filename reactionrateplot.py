import matplotlib.pyplot as plt
import numpy as np

menke_2015 = [[8.8e-5,4.6e-5,2.3e-5],[0,67,149]] #rxn rate (mol/m2s), time (min)
menke_2017_ketton_36 = [[6.71e-5,6.74e-5,3.27e-5,3.76e-5],[0,60,120,240]] #rxn rate (mol/m2s), time (min)
menke_2017_ketton_31 = [[4.57e-5,1.71e-5,1.78e-5,8.71e-6],[0,60,120,240]] #rxn rate (mol/m2s), time (min)
menke_2017_estaillad = [[1.85e-5,1.07e-5,4.26e-6,4.26e-6],[0,60,120,240]] #rxn rate (mol/m2s), time (min)
menke_2017_portland_ = [[2.71e-5,1.44e-5,3.57e-6,1.16e-6],[0,60,120,240]] #rxn rate (mol/m2s), time (min)

data_sets = [menke_2015,menke_2017_ketton_36,menke_2017_ketton_31,menke_2017_estaillad,menke_2017_portland_]
data_labels = ['menke_2015','menke_2017_ketton_36','menke_2017_ketton_31','menke_2017_estaillades','menke_2017_portland']

fig, ax = plt.subplots()
for ind,set in enumerate(data_sets):
    ax.plot(set[1],set[0],label = data_labels[ind])

ax.tick_params(axis='both',labelsize=14)
ax.set_xlabel('time [min]', fontsize=15)
ax.set_ylabel('reaction rate [mol/m^2s]',fontsize=15)
ax.legend()
plt.tight_layout()
plt.show()
