import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px

df = pd.read_csv('flow_transport_rxn_properties.csv',header=0)
df.drop(3,axis=0,inplace=True)

heterogeneity = "EMD"
deg = 2

X = np.column_stack([df[heterogeneity].values.tolist(),df['Pe'].values.tolist(),df['Da'].values.tolist()])
Y = df['ratio'].values.tolist()

poly_features = PolynomialFeatures(degree=deg)
lin_reg = LinearRegression()

pipeline = Pipeline([("polynomial_features", poly_features), ("linear_regression", lin_reg)])
pipeline.fit(X, Y)

y_pred = pipeline.predict(X)
mean = np.mean(Y)
r2 = 1 - (np.sum((y_pred-Y)**2))/(np.sum((Y-mean)**2))
print(r2)
# print(pipeline.named_steps['linear_regression'].intercept_,pipeline.named_steps['linear_regression'].coef_,len(pipeline.named_steps['linear_regression'].coef_))
w0 = pipeline.named_steps['linear_regression'].intercept_
w = pipeline.named_steps['linear_regression'].coef_
H = np.linspace(np.min(X[:,0]),np.max(X[:,0]),100)
Pe = np.linspace(np.min(X[:,1]),np.max(X[:,1]),100)
Da = np.linspace(np.min(X[:,2]),np.max(X[:,2]),100)
rxn = w0 + w[1]*H+w[2]*Pe+w[3]*Da+w[4]*H*Pe+w[5]*H*Da+w[6]*Pe*Da+w[7]*H**2+w[8]*Pe**2+w[9]*Da**2

pca = PCA(1)
pca.fit(np.column_stack([Pe,Da]))
print(pca.components_,pca.explained_variance_)
new = pca.transform(np.column_stack([Pe,Da]))

data = {'H':H,'Pe':Pe,'Da':Da,'rxn':rxn}
df2 = pd.DataFrame(data=data)

fig = px.scatter_3d(df2,x='Pe',y='Da',z='H',color='rxn')
fig.show()
# print(sklearn.metrics.SCORERS.keys())
# scores = cross_val_score(pipeline, X, Y, scoring="neg_mean_absolute_error", cv=10)

# print(scores)