import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plot = False

df = pd.read_csv('glm_results.csv',index_col=False,header=None,nrows=3).T
df.columns = ['eqn','link','aic']

for row in range(len(df.aic)):
    if len(df.aic.values[row]) > 7:
        df.aic[row] = df.aic.values[row][0:5]
    df.aic[row] = float(df.aic.values[row])

df.drop(np.where(df.aic >4000)[0],inplace=True)
df.sort_values('aic',axis=0,inplace=True,ascending=False)
print(df[40:60])

if plot == True:
    aics_plot = df.aic[df.aic <4000].values
    fig,ax = plt.subplots()
    ax.grid(axis='y',zorder=0)
    ax.hist(aics_plot,bins=50,facecolor='blue',edgecolor='navy',zorder=3)
    ax.set_xlabel('AIC',fontsize=15)
    ax.set_ylabel('Count',fontsize=15)
    ax.tick_params(labelsize=13)
    ax.set_yticks(np.arange(0,22,2))
    fig.tight_layout()
    plt.show()