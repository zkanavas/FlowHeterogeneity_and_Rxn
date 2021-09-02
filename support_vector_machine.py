# build my SVM model
from numpy.core.fromnumeric import std
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score,r2_score,explained_variance_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def plot_hyperplane(x,z,behavior,svc):
    fig,ax = plt.subplots()
    #create grid
    xmin, xmax = np.min(x),np.max(x)
    zmin, zmax = np.min(z),np.max(z)
    # bmin, bmax = np.min(behavior),np.max(behavior)
    xx = np.linspace(xmin,xmax,100)
    zz = np.linspace(zmin,zmax,100)
    ZZ,XX = np.meshgrid(zz,xx)
    xz = np.vstack([XX.ravel(),ZZ.ravel()]).T
    colors = ['red' if beh == 'uniform' else 'blue' for beh in behavior]
    ax.scatter(x,z,c=colors)
    B = svc.decision_function(xz).reshape(XX.shape)
    ax.contour(xx,zz,B,levels=[-1,0,1],colors='k',linestyles=['--','-','--'])
    plt.show()



crossval = False
randomstates = np.random.randint(1,500,100)
randomstate=1
heterogeneities = ['pc','EMD']#'Mm','EMD']
df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)
df.drop(3,axis=0,inplace=True)

#what I want to predict
behavior = df['behavior'].values

#attributes
x = df['Pe'].values.tolist()
y = df['Da'].values.tolist()

for heterogeneity in heterogeneities:
    z = df[heterogeneity].values.tolist()
    # scores  = []
    # for randomstate in randomstates:

    scaler = StandardScaler()
    if heterogeneity == 'pc':
        svc = SVC(kernel = 'poly',C=1,gamma=1,degree=1,coef0=0)
        scaler.fit(np.column_stack([x,z]))
        attributes = scaler.transform(np.column_stack([x,z]))
    elif heterogeneity == 'Mm':
        svc = SVC(kernel='sigmoid',C=10,gamma=10,coef0=10)
        scaler.fit(np.array(z).reshape(-1, 1))
        attributes = scaler.transform(np.array(z).reshape(-1, 1))
    elif heterogeneity == 'EMD':
        svc = SVC(kernel = 'rbf',C=10,gamma=1)
        scaler.fit(np.column_stack([x,z]))
        attributes = scaler.transform(np.column_stack([x,z]))

    if crossval == True:
        #cross validation
        scores = []
        kf = KFold(n_splits=len(df),shuffle=True,random_state=randomstate)
        for train_index, test_index in kf.split(attributes):
            att_train, att_test = attributes[train_index],attributes[test_index]
            beh_train, beh_test = behavior[train_index], behavior[test_index]
        
            svc.fit(att_train,beh_train)

            scores.append(svc.score(att_test,beh_test))
            print(svc.dual_coef_)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(heterogeneity,mean_score,std_score)
    else:
        svc.fit(attributes,behavior)
        score = svc.score(attributes,behavior)
        plot_hyperplane(attributes[:,0],attributes[:,1],behavior,svc)
        print(heterogeneity, score)


