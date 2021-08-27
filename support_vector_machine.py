# build my SVM model
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score,r2_score,explained_variance_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

randomstates = np.random.randint(1,500,100)
randomstate=1
heterogeneities = ['pc','Mm','EMD']
parameters = {'kernel':('linear', 'rbf','poly','sigmoid'), 'C':[1,10,100,1000],'gamma':[0.0001,0.001,0.01,0.1,0.2,.3],'degree':[2,3,4]}
df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)
df.drop(3,axis=0,inplace=True)

#what I want to predict
behavior = df['behavior'].values.tolist()
rxnratio = df['ratio'].values.tolist()

#attributes
x = df['Pe'].values.tolist()
y = df['Da'].values.tolist()

for heterogeneity in heterogeneities:
    z = df[heterogeneity].values.tolist()
    # scores  = []
    # for randomstate in randomstates:
    X,Y,Z,B,R = shuffle(x,y,z,behavior,rxnratio, random_state = randomstate)
    x_train, x_test, y_train, y_test, z_train, z_test, B_train, B_test, r_train, r_test = train_test_split(X, Y, Z, B, R, train_size=0.9,random_state = randomstate)
    svc_ = SVC()
    clf = GridSearchCV(svc_,parameters,cv=2)
    #1-d
    scaler = StandardScaler()
    scaler.fit(np.array(z_train).reshape(-1, 1))
    attributes = scaler.transform(np.array(z_train).reshape(-1, 1))
    clf.fit(attributes,B_train)
    print('1d',heterogeneity,clf.best_score_,clf.best_params_)
    
    #2-d
    scaler = StandardScaler()
    scaler.fit(np.column_stack([x_train,z_train]))
    attributes = scaler.transform(np.column_stack([x_train,z_train]))
    clf.fit(attributes,B_train)
    print('2d-Pe',heterogeneity,clf.best_score_,clf.best_params_)
    
    #2-d
    scaler = StandardScaler()
    scaler.fit(np.column_stack([y_train,z_train]))
    attributes = scaler.transform(np.column_stack([y_train,z_train]))
    clf.fit(attributes,B_train)
    print('2d-Da',heterogeneity,clf.best_score_,clf.best_params_)

    #3-d
    scaler.fit(np.column_stack([x_train,y_train,z_train]))
    attributes = scaler.transform(np.column_stack([x_train,y_train,z_train]))
    clf.fit(attributes,B_train)
    print('3d',heterogeneity,clf.best_score_,clf.best_params_)

    #     scores.append(clf.best_score_)
    # print(heterogeneity, " mean score: ", np.mean(scores), " score std: ", np.std(scores))
        # clf = make_pipeline(StandardScaler(),SVC(kernel=kernel))
        # # clf = SVC(kernel=kernel,verbose=True)
        # clf.fit(np.column_stack([x_train,y_train,z_train]),B_train)
        # B_pred = clf.predict(np.column_stack([x_test,y_test,z_test]))
        # print(kernel,heterogeneity, " self score: ", clf.score(np.column_stack([x_train,y_train,z_train]),B_train))
        # print(kernel,heterogeneity, " predicted accuracy: ",accuracy_score(B_test,B_pred))

        # reg = make_pipeline(StandardScaler(),SVR(kernel=kernel))
        # # reg = SVR(kernel = kernel,verbose=True)
        # reg.fit(np.column_stack([x_train,y_train,z_train]),r_train)
        # r_pred = reg.predict(np.column_stack([x_test,y_test,z_test]))
        # print(kernel,heterogeneity, " self score: ", reg.score(np.column_stack([x_train,y_train,z_train]),r_train))
        # print(kernel,heterogeneity, " r2: ", r2_score(r_test,r_pred))
        # print(explained_variance_score(r_test,r_pred))

# clf = make_pipeline(StandardScaler(),SVC(kernel=kernel))
# # clf = SVC(kernel=kernel,verbose=True)
# clf.fit(np.column_stack([X,Y,Z]),B)
# print(clf.score(np.column_stack([X,Y,Z]),B))

# reg = make_pipeline(StandardScaler(),SVR(kernel=kernel))
# # reg = SVR(kernel = kernel,verbose=True)
# reg.fit(np.column_stack([X,Y,Z]),R)
# print(reg.score(np.column_stack([X,Y,Z]),R))

