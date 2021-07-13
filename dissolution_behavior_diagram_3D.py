#dissolution behavior diagram
import pandas as pd
import plotly.express as px

d = {'Pe':[100,100],'Da':[1.8e-4,4.0e-4],'Mm':[6.81,8.66],'datadescriptor':['beadpack','estaillades']}
df = pd.DataFrame(data=d)

fig = px.scatter_3d(df, x='Pe', y='Da', z='Mm',
              color='datadescriptor')
fig.show()