# build my SVM model
from numpy import random
from numpy.lib.shape_base import column_stack
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score,r2_score,explained_variance_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

randomstates = np.random.randint(1,500,100)
randomstate=1
d=1
heterogeneities = ['EMD']#,'Mm','EMD']
# heterogeneity = 'Mm'
kernels = ['linear']#,'rbf','sigmoid','poly']


df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)
df.drop(3,axis=0,inplace=True)

#what I want to predict
behavior = np.array(df['behavior'].values.tolist())
rxnratio = np.array(df['ratio'].values.tolist())

#attributes
x = df['Pe'].values.tolist()
y = df['Da'].values.tolist()

for heterogeneity in heterogeneities:
    
    z = df[heterogeneity].values.tolist()
    X = np.column_stack([x,y,z])
    # X = np.array(z).reshape(-1,1)
   
    #scale attributes and target to have var=1 mean~=0
    scaler = StandardScaler()
    scaler.fit(X)
    attributes = scaler.transform(X)

    attributes, B, R = shuffle(attributes,behavior,rxnratio,random_state=randomstate)
    att_train,att_test,beh_train,beh_test,rxn_train,rxn_test = train_test_split(attributes,B,R,train_size=0.9,random_state=randomstate)
    # for randomstate in randomstates:
    # X,Y,Z,B,R = shuffle(x,y,z,behavior,rxnratio, random_state = randomstate)
    # x_train, x_test, y_train, y_test, z_train, z_test, B_train, B_test, r_train, r_test = train_test_split(X, Y, Z, B, R, train_size=0.9,random_state = randomstate)


    #cross validation
    # kf = KFold(n_splits=len(df),shuffle=True,random_state=randomstate)
    # for train_index, test_index in kf.split(attributes):
    #     att_train, att_test = attributes[train_index],attributes[test_index]
    #     rxn_train, rxn_test = rxnratio[train_index], rxnratio[test_index]
    #     beh_train, beh_test = behavior[train_index], behavior[test_index]

    # for kernel in kernels:
    #     if kernel =='linear':
    #         parameters = {'C':[1e-1,1,10,100],'tol':[1e-5,1e-3,1e-2]} 
    #     if kernel == 'rbf':
    #         parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10]}
    #     if kernel == 'sigmoid':
    #         parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10],'coef0':[0,1,10]}
    #     if kernel =='poly':
    #         parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10],'degree':[2,3,4],'coef0':[0,1,10]}
    kernel = 'poly'
    svr = SVR(kernel = kernel,epsilon=0.0001,degree=2,C=10)
    svr.fit(att_train,rxn_train)
    rxn_pred = svr.predict(att_test)
    print(svr.score(att_test,rxn_test))
    # mean = np.mean(rxn_test)
    # r2 = 1 - (np.sum((rxn_pred-rxn_test)**2))/(np.sum((rxn_test-mean)**2))
    print(rxn_pred,rxn_test)

        