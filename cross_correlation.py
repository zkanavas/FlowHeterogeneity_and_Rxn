import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0,index_col=0)
df.drop('menke_ketton',inplace=True)
df.drop(['nx','ny','nz','rb','Vol_perc','SA_perc','SSA_perc','Mm','sigma_v','mu_v'],inplace=True,axis=1)

#reorder dataframe columns
cols = ['ratio','Pe','Da','EMD','pc','Vol_hv','SA_hv','SSA']
col_labels = ['ratio','Pe','Da','EMD',r'$p_c$',r'$Vol_{HVR}$',r'$SA_{HVR}$','SSA']

newdf = df[cols]
#compute correlation matrix
corr = newdf.corr(method='spearman')
pvalues = calculate_pvalues(newdf) 

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))
# mask_p = np.triu(np.ones_like(pvalues, dtype=bool))
# corr = corr.where(np.tril(np.ones_like(corr, dtype=bool)))
pvalues = pvalues.where(np.tril(np.ones_like(pvalues, dtype=bool)))

print(corr,pvalues)

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, center=0, vmin= -1, vmax = 1,
            square=True, linewidths=.5, 
            cbar_kws={"shrink": .5,'label':r'Spearman $\rho$','orientation':'horizontal'},
            annot=True, annot_kws={"fontsize":14},ax=ax)

ax.tick_params('both',labelsize = 15)
# ylabels = ax.get_yticklabels()
ax.set_yticklabels(col_labels,rotation = 'horizontal')
ax.set_xticklabels(col_labels)

f.tight_layout()
plt.show()
