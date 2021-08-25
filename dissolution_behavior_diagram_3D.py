#dissolution behavior diagram
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

plot = True
df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)

if plot == True:
    fig = px.scatter_3d(df, x='Pe', y='Da', z='EMD',
                color='behavior')
    # fig.write_html("behaviordiagram_Mm_behavior.html")
    fig.show()

# X = np.vstack([df.Pe.values,df.Da.values,df.Mm.values]).transpose()
# Y = df.behavior.values

# logReg = LogisticRegression(solver = 'lbfgs')
# logReg.fit(X,Y)

# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, 3500#X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# z_min, z_max = X[:, 2].min() - .5, X[:, 2].max() + .5

# h = 1  # step size in the mesh
# xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 100), np.arange(y_min, y_max, 0.05),np.arange(z_min, z_max, 0.1))
# # Z = logReg.predict(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])
# # Z[Z=='uniform'] ='red'
# # Z[Z=='wormhole'] ='blue'

# fig, ax = plt.subplots()
# colors = df.behavior.values
# colors[colors=='uniform'] ='red'
# colors[colors=='wormhole'] ='blue'
# ax.scatter(df.Pe.values,df.Mm.values,c=colors)
# z = xx.ravel()*(-0.12/300)+5.5
# ax.plot(xx.ravel(),z,'g-')
# ax.tick_params(axis='both',labelsize=14)
# ax.set_xlabel('Pe', fontsize=15)
# ax.set_ylabel('Mm',fontsize=15)
# plt.tight_layout()
# plt.show()

# data = {'Pe':xx.ravel(),'Da':yy.ravel(),'Mm':zz.ravel(),'behavior':Z}
# df2 = pd.DataFrame.from_dict(data)
# fig = px.scatter_3d(df, x='Pe', y='Da', z='Mm',
#                 color='behavior')
# fig.update_traces(marker=dict(size=5))
# fig.add_traces(go.Surface(x=xx, y=yrange, z=pred, name='pred_surface'))
# fig.show()

# fig = go.Figure(data=[go.Scatter3d(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),opacity = 0.9, mode='markers',marker=dict(color=Z))])
# fig.show()
# fig = go.Figure(data=[go.Mesh3d(x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),colorscale = [[0, 'red'],[1,'blue']],intensity = Z, opacity=0.50)])
# fig.show()