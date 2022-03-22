#dissolution behavior diagram
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

plot_golfier = False

# directory = r"C:\Users\zkanavas\Pictures"
directory = r"C:\Users\zkana\Pictures"

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
# df.drop(3,axis=0,inplace=True)
df.drop(2,axis=0,inplace=True)

golfier_uniform = pd.read_csv('golfier_uniform.csv',header=None,names=["Pe","Da"])
golfier_wormhole = pd.read_csv('golfier_wormhole.csv',names=["Pe","Da"])
golfier_compact = pd.read_csv('golfier_compact.csv',names=["Pe","Da"])

# df.drop([3,12,13,14,15],axis=0,inplace=True)
# x = df.Mm.values * df.EMD.values
# if plot == True:

# size_min = 10
# size_max = 1000
# df.ratio = ((df.ratio - np.min(df.ratio))/np.ptp(df.ratio))*(size_max-size_min)+size_min
colors = []
for color in df.behavior:
    if color == "wormhole":
        colors.append('blue')
    elif color == "uniform":
        colors.append('red')
    elif color == "compact":
        colors.append('green')
size_min = 10
size_max = 100
scaled_ratio = ((df.ratio - np.min(df.ratio))/np.ptp(df.ratio))*(size_max-size_min)+size_min

fig = go.Figure(data=[go.Scatter3d(
                                    x=df.pc,
                                    y=df.Pe,
                                    z=df.adv_Da,
                                    hovertext=df.Sample_Name,
                                    mode = 'markers',
                                    marker = dict(
                                                    # size=scaled_ratio,
                                                    color=colors
                                    ))])                                    
fig.add_trace(go.Scatter3d(
                                    x=np.ones(len(golfier_uniform)),
                                    y=golfier_uniform.Pe,
                                    z=golfier_uniform.Da,
                                    hovertext = "Golfier Uniform",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'red'
                                    )))
fig.add_trace(go.Scatter3d(
                                    x=np.ones(len(golfier_wormhole)),
                                    y=golfier_wormhole.Pe,
                                    z=golfier_wormhole.Da,
                                    hovertext = "Golfier Wormhole",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'blue'
                                    )))     
fig.add_trace(go.Scatter3d(
                                    x=np.ones(3),
                                    y=golfier_compact.Pe,
                                    z=golfier_compact.Da,
                                    hovertext = "Golfier Compact",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'green'
                                    )))      
fig.add_trace(go.Scatter3d(
                            x = [1,1],
                            y = [0.0001,10000],
                            z = [0.00042,0.00042],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5
                            )))
fig.add_trace(go.Scatter3d(
                            x = [1,1],
                            y = [0.0015,0.0015],
                            z = [0.00042,1000],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5
                            )))                            
fig.add_trace(go.Scatter3d(
                            x=[1,1,1],
                            y=[0.0002,3,3],
                            z=[2,0.0001,2],
                            text=["Compact","Uniform","Wormhole"],
                            textposition="middle center",
                            mode="text"))                           
fig.add_trace(go.Scatter3d(
                            x=[1,5.5],
                            y=[50,10000],
                            z=[0.0000001,0.0000001],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5)))   
fig.add_trace(go.Scatter3d(
                            x=[1,10],
                            y=[0.0015,0.0015],
                            z=[0.0000001,0.0000001],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5)))                                                           
fig.add_trace(go.Scatter3d(
                            x=[4.5,2,8],
                            y=[0.0002,1000,100],
                            z=[0.0000002,0.0000002,0.0000002],
                            text=["Compact","Uniform","Wormhole"],
                            textposition="middle center",
                            mode="text")) 
x_pc = [8.909,8.909,8.909,6.742,6.742,6.742,2.843,2.843,2.843]
y_Pe = [5e-4,0.5,50,5e-4,0.5,50,5e-4,0.5,50]
z_Da = [120.4,0.1204,0.001204,68.64,0.06864,0.0006864,134.2,.1342,0.001342]
fig.add_trace(go.Scatter3d(
                            x=x_pc,y=y_Pe,z=z_Da,
                            mode="markers",
                            hovertext = "new sim?",
                            marker=dict(
                                color="orange"
                            )
))
fig.update_layout(scene = dict(
                    xaxis_title='pc',
                    yaxis_title='Pe',
                    zaxis_title='adv Da',
                    yaxis=dict(type='log',range=[-4,4],),
                    zaxis=dict(type='log',range=[-7,3])))                                  

# fig.write_html(directory+"\Pe_advDa_pc_golfier_proposedpoints.html")
# fig.show()

fig = go.Figure(data=[go.Scatter3d(
                                    x=df.pc,
                                    y=df.Pe,
                                    z=df.diff_Da,
                                    hovertext=df.Sample_Name,
                                    mode = 'markers',
                                    marker = dict(
                                                    # size=scaled_ratio,
                                                    color=colors
                                    ))])                                    
fig.add_trace(go.Scatter3d(
                                    x=np.ones(len(golfier_uniform)),
                                    y=golfier_uniform.Pe,
                                    z=golfier_uniform.Pe*golfier_uniform.Da,
                                    hovertext = "Golfier Uniform",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'red'
                                    )))
fig.add_trace(go.Scatter3d(
                                    x=np.ones(len(golfier_wormhole)),
                                    y=golfier_wormhole.Pe,
                                    z=golfier_wormhole.Pe*golfier_wormhole.Da,
                                    hovertext = "Golfier Wormhole",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'blue'
                                    )))     
fig.add_trace(go.Scatter3d(
                                    x=np.ones(3),
                                    y=golfier_compact.Pe,
                                    z=golfier_compact.Pe*golfier_compact.Da,
                                    hovertext = "Golfier Compact",
                                    mode="markers",
                                    marker = dict(
                                                    color = 'green'
                                    )))      
fig.add_trace(go.Scatter3d(
                            x = [1,1],
                            y = [0.0001,10000],
                            z = [0.0000001,10],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5
                            )))
fig.add_trace(go.Scatter3d(
                            x = [1,1],
                            y = [0.0015,0.0015],
                            z = [0.000001,1000],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5
                            )))                            
fig.add_trace(go.Scatter3d(
                            x=[1,1,1],
                            y=[0.0002,10,.3],
                            z=[0.001,0.001,1],
                            text=["Compact","Uniform","Wormhole"],
                            textposition="middle center",
                            mode="text"))                           
fig.add_trace(go.Scatter3d(
                            x=[1,5.5],
                            y=[50,10000],
                            z=[0.1,0.1],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5)))   
fig.add_trace(go.Scatter3d(
                            x=[1,10],
                            y=[0.0015,0.0015],
                            z=[0.1,0.1],
                            mode="lines",
                            line=dict(
                                color='black',
                                width=5)))                                                           
fig.add_trace(go.Scatter3d(
                            x=[4.5,2,8],
                            y=[0.0002,1000,100],
                            z=[0.2,0.2,0.2],
                            text=["Compact","Uniform","Wormhole"],
                            textposition="middle center",
                            mode="text")) 
x_pc = [8.909,8.909,8.909,6.742,6.742,6.742,2.843,2.843,2.843]
y_Pe = [5e-4,0.5,50,5e-4,0.5,50,5e-4,0.5,50]
z_Da = [120.4,0.1204,0.001204,68.64,0.06864,0.0006864,134.2,.1342,0.001342]
# fig.add_trace(go.Scatter3d(
#                             x=x_pc,y=y_Pe,z=np.multiply(y_Pe,z_Da),
#                             mode="markers",
#                             hovertext = "new sim?",
#                             marker=dict(
#                                 color="orange"
#                             )
# ))
fig.update_layout(scene = dict(
                    xaxis_title='pc',
                    yaxis_title='Pe',
                    zaxis_title='K',
                    yaxis=dict(type='log',range=[-4,4],),
                    zaxis=dict(type='log',range=[-6,1])))      
fig.write_html(directory+"\Pe_diffDa_pc_golfier_v1.html")

fig.show()

fig = px.scatter_3d(df, x='pc', y='Pe', z='diff_Da',size='ratio',size_max=100,
            color='behavior',hover_name='Sample_Name')

# fig.write_html(directory+"\Pe_diffDa_pc_ratio_beh_all.html")

# fig.show()
fig = px.scatter_3d(df, x='pc', y='Pe', z='adv_Da',size='ratio',size_max=100,
            color='behavior',log_z=True)
# fig.write_html(directory+"\Pe_advDa_pc_ratio_beh_all.html")

# fig.show()



# behavior = df.behavior.values
# colors = ['red' if beh == 'uniform' else 'blue' for beh in behavior]
# x = df.Mm.values
# y=df.EMD.values
# z = df.pc.values
# rxn = df.ratio.values
# y_trans = np.log(y)
# # y_trans = np.exp(y)/(1+np.exp(y))
# fig, ax = plt.subplots()
# x_trans = np.exp(x)/(1+np.exp(x))

# ax.scatter(y_trans*x_trans,rxn,c=colors)


#golfier diagram
if plot_golfier==True:
    Pe = df.Pe
    Da = df.Da
    behavior = df.behavior.values
    markers = ['P' if beh == 'uniform' else 'o' for beh in behavior]
    chemical = df.chemical_makeup
    colors = []
    for chem in chemical:
        if chem == 0:
            colors.append('red')
        elif chem == 1:
            colors.append('blue')
        elif chem == 2:
            colors.append('limegreen')
        else: colors.append('purple')

    fig, ax = plt.subplots()

    for i in range(len(Pe)):
        ax.scatter(Pe[i], Da[i]*Pe[i], c=colors[i], marker=markers[i])
    # ax.axhline(y=3e-3,c='k')
    ax.xaxis.tick_top()
    ax.set_xlabel('Pe')    
    # ax.set_xlim([5e1,1e4])
    ax.set_xlim([1e-4,1e4])
    ax.xaxis.set_label_position('top') 
    ax.invert_yaxis()
    ax.set_ylabel('Pe*Da')
    # ax.set_ylim([2e-3,1e-6])
    ax.set_ylim([1e1,1e-7])
    # ax.axvline(x=9e-3,ymin=0,ymax=0.6125,c='k')
    ax.plot([1e4,1.25e-4],[8,1e-7],'-',c='k')
    ax.axvline(x=9e-3,ymin=0,ymax=0.77,c='k')
    ax.loglog()
    plt.show()

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