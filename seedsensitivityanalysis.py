import pandas as pd
import matplotlib.pyplot as plt

sensitivityresults = pd.read_csv('formattedseedsenssol.csv',header=0)
sensitivityresults.drop(['Pe','MmPe','pcPe'],axis=1,inplace=True)
cols = sensitivityresults.columns.tolist()
removecols = ['seed','aic','r2','aic_quantile','r2_quantile']
[cols.remove(col) for col in removecols]
fig, ax = plt.subplots()

pd.plotting.parallel_coordinates(sensitivityresults.sort_values('r2_quantile',ascending=False),'r2_quantile',cols=cols,color = ['red','yellow','green'],ax=ax,alpha=0.5)
# pd.plotting.parallel_coordinates(sensitivityresults.sort_values('aic_quantile',ascending=False),'aic_quantile',cols=cols,color = ['red','yellow','green'],ax=ax[1],alpha=0.5)
ax.set_ylabel('Coefficient Value',fontsize=15)
ax.tick_params('both',labelsize=15)
ax.legend(fontsize=14,loc='upper left',bbox_to_anchor=(1, 1))
fig.tight_layout()

plt.show()

# index = sensitivityresults.aic.idxmin()
# print(sensitivityresults.iloc[index])

# #stats

# nonzero_mean = sensitivityresults.numbernonzeros.mean()
# r2_mean = sensitivityresults.r2.mean()
# aic_mean = sensitivityresults.aic.mean()

# nonzero_std = sensitivityresults.numbernonzeros.std()
# r2_std = sensitivityresults.r2.std()
# aic_std = sensitivityresults.aic.std()

# solutions = sensitivityresults.solution.str.translate({ ord(c): None for c in "\n[]" })
# # solutions = solutions.str.replace(". ",".,")
# solutions = solutions.str.replace("        ","")
# solutions = solutions.str.replace(" ",",")
# for ind,solution in enumerate(solutions):
#     solutions[ind] = list(solution.split(","))
# for i, solution in enumerate(solutions):
#     for ind,ele in enumerate(solution):
#         if ele == "":
#             solutions[i].pop(ind)
# for i, solution in enumerate(solutions):
#     for ind,ele in enumerate(solution):
#         solution[ind] = float(ele)
#     solutions[i] = solution

# print('hi')
