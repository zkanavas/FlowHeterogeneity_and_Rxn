from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def svc(heterogeneity,df,randomstate,kernel,C,gamma,coef0,deg,plot=True):
    #what I want to predict
    behavior = df['behavior'].values.tolist()
    rxnratio = df['ratio'].values.tolist()

    #attributes
    x = df['Pe'].values.tolist()
    y = df['Da'].values.tolist()
    z = df[heterogeneity].values.tolist()
    X,Y,Z,B,R = shuffle(x,y,z,behavior,rxnratio, random_state = randomstate)
    x_train, x_test, y_train, y_test, z_train, z_test, B_train, B_test, r_train, r_test = train_test_split(X, Y, Z, B, R, train_size=0.9,random_state = randomstate)
    clf = SVC(kernel = kernel,C=C,gamma=gamma,coef0=coef0,degree=deg)
    scaler = StandardScaler()
    scaler.fit(np.column_stack([x_train,z_train]))
    attributes = scaler.transform(np.column_stack([x_train,z_train]))
    clf.fit(attributes,B_train)
    print(heterogeneity, " self score: ", clf.score(attributes,B_train))
    print(clf.dual_coef_,clf.intercept_,clf.n_support_)
    if plot == True:
        # plot_hyperplane(clf,X,Z,B,scaler)
        plot_contour(clf,X,Z,B,scaler)
    return clf

def plot_contour(clf,X,Z,B,scaler):
    fig,ax = plt.subplots()
    #plot datapoints and color according to their class
    color = ['black' if c == 'uniform' else 'lightgrey' for c in B]
    attributes = scaler.transform(np.column_stack([X,Z]))
    ax.scatter(attributes[:,0],attributes[:,1],c=color)
    #plot decision surface 
    x_min,x_max = np.min(attributes[:,0]), np.max(attributes[:,0])
    z_min,z_max = np.min(attributes[:,1]), np.max(attributes[:,1])
    xx,zz = np.meshgrid(np.arange(x_min,x_max,0.2),np.arange(z_min,z_max,0.2))
    b_pred = clf.predict(np.column_stack([xx.ravel(),zz.ravel()]))
    b_pred = np.array([1 if b == 'uniform' else -1 for b in b_pred])
    b_pred = b_pred.reshape(xx.shape)
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(B))])
    ax.contour(xx,zz,b_pred,alpha=0.4,cmap=cmap)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(zz.min(), zz.max())
    plt.show()

def plot_hyperplane(clf,X,Z,B,scaler):
    fig,ax = plt.subplots()
    #plot datapoints and color according to their class
    color = ['black' if c == 'uniform' else 'lightgrey' for c in B]
    attributes = scaler.transform(np.column_stack([X,Z]))
    ax.scatter(attributes[:,0],attributes[:,1],c=color)
    #create hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(np.min(attributes[:,0]),np.max(attributes[:,0]))
    # zz = np.linspace(np.min(z_train),np.max(z_train))
    zz = a * xx - (clf.intercept_[0]) / w[1]
    #plot hyperplane
    ax.plot(xx,zz)
    plt.show()


randomstate=1

df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)
# df.drop(3,axis=0,inplace=True)

pc_clf = svc('pc',df,randomstate,'linear',10,'auto',0,0,plot=True)
Mm_clf = svc('Mm',df,randomstate,'sigmoid',10,1,0,0,plot=True)
EMD_clf = svc('EMD',df,randomstate,'rbf',10,1,0,0,plot=True)