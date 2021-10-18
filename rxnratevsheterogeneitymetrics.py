import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df = df.sort_values(by='pc')

fig,(ax1,ax2,ax3) = plt.subplots(1,3,sharey = True,figsize=(12,6))
for index, row in df.iterrows():
    # if row['Sample_Name'] == 'menke_ketton': continue
    ax1.scatter(row['pc'],row['ratio'],label = row['Sample_Name'])
    ax2.scatter(row['Mm'],row['ratio'],label = row['Sample_Name'])
    ax3.scatter(row['EMD'],row['ratio'],label = row['Sample_Name'])
ax3.semilogx()
# ax1.semilogy()
# ax2.semilogy()
ax1.set_ylabel("Rxn Ratio",fontsize=15)
ax1.set_xlabel("pc",fontsize=15)
ax1.tick_params('both', labelsize= 14)

ax2.set_xlabel("Mm",fontsize=15)
ax2.tick_params('x',labelsize=14)

ax3.set_xlabel("EMD",fontsize=15)
ax3.tick_params('x',labelsize=14)
ax3.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize =14)

plt.tight_layout()
plt.show()
