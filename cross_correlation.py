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
# df.drop(['nx','ny','nz','rb','Vol_perc','SA_perc','SSA_perc','Mm','sigma_v','mu_v'],inplace=True,axis=1)

#reorder dataframe columns
cols = ['ratio','Pe','adv_Da','diff_Da','EMD_norm','EMD_log','pc','Vol_hv','SA_hv','chemical_makeup','chemical_heterogeneity']
col_labels = ['ratio','Pe',r'$Da_{adv}$',r'$Da_{diff}$',r'$EMD_{norm}$',r'$EMD_{log}$',r'$p_c$','V','S',r'$che_{min}$',r'$che_{het}$']
# cols = ['ratio','Vol_hv','Vol_perc','SA_hv','SA_perc','SSA','SSA_perc']
# col_labels = ['ratio','V_hv','V_perc','S_hv','S_perc','SSA','SSA_perc']
newdf = df[cols]
#compute correlation matrix
corr = newdf.corr(method='spearman')
pvalues = calculate_pvalues(newdf) 

# Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
# mask_p = np.triu(np.ones_like(pvalues, dtype=bool))
# corr = corr.where(np.tril(np.ones_like(corr, dtype=bool)))
# pvalues = pvalues.where(np.tril(np.ones_like(pvalues, dtype=bool)))

print(corr,pvalues)

mask = np.ones_like(corr,dtype=bool)
for i,col in enumerate(pvalues.columns):
    for j,ind in enumerate(pvalues.index):
        if pvalues[col][ind] <= 0.01: mask[j,i] = False

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(150, 275, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, center=0, mask = np.invert(mask), vmin= -1, vmax = 1,
            square=True, linewidths=.5, 
            cbar = False,
            # cbar_kws={"shrink": .5,'label':r'Spearman $\rho$','orientation':'horizontal'},
            annot=True, annot_kws={"fontsize":14},ax=ax)
sns.heatmap(corr, cmap=cmap, mask = mask,center=0, vmin= -1, vmax = 1,
            square=True, linewidths=.5, 
            cbar = False,
            annot=True, annot_kws={"fontsize":14,"weight":"bold"},ax=ax)
ax.tick_params('both',labelsize = 15)
# ylabels = ax.get_yticklabels()
ax.set_yticklabels(col_labels,rotation = 'horizontal')
ax.set_xticklabels(col_labels,rotation = 'horizontal')

f.tight_layout()
plt.show()
