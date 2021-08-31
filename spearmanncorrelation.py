from scipy.stats import spearmanr
import pandas as pd
import matplotlib.pyplot as plt

heterogeneities = ["pc","Mm","EMD"]
df = pd.read_csv('flow_transport_rxn_properties.csv',header = 0)

for heterogeneity in heterogeneities:
    corr, pvalue = spearmanr(df[heterogeneity],df['ratio'],axis=0)
    print(heterogeneity, "correlation: ",corr, " p-value: ",pvalue)
