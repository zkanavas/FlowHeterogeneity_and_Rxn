#generalized linear regression
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
import numpy as np

randomstate = 1

df = pd.read_csv('flow_transport_rxn_properties.csv', header = 0)

# heterogeneities = ['pc','Mm','EMD']
heterogeneity = ['EMD']

#what I want to predict
rxnratio = np.array(df['ratio'].values.tolist())

#attributes
x = df['Pe'].values.tolist()
y = df['Da'].values.tolist()

# for heterogeneity in heterogeneities:
z = df[heterogeneity].values.tolist()

X = np.column_stack([x,y,z])

#scale attributes to have var=1 mean~=0
scaler = StandardScaler()
scaler.fit(X)
attributes = scaler.transform(X)

parameters = {'alpha':[10,100,1]}

#cross validation
kf = KFold(n_splits=len(df),shuffle=True,random_state=randomstate)
for train_index, test_index in kf.split(attributes):
    att_train, att_test = attributes[train_index],attributes[test_index]
    target_train, target_test = rxnratio[train_index], rxnratio[test_index]
    #initialize predictive model
    glm = TweedieRegressor(max_iter=1000,power=3,link='auto')
    reg = GridSearchCV(glm,parameters)
    # reg = TweedieRegressor(power = 0, alpha=1,link='auto',max_iter=100,tol=1e-4)
    #fit predictive model
    reg.fit(att_train,target_train)
    print(reg.best_score_,reg.best_params_)
    #predict test sample
    pred = reg.predict(att_test)
    #evaulate prediction
    score = (target_test-pred)/target_test
    print(score)