#dissolution behavior diagram
import pandas as pd
import plotly.express as px

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)

fig = px.scatter_3d(df, x='Pe', y='Da', z='pc',
              color='ratio')
fig.show()
# fig.write_html("behaviordiagram_pc_names.html")