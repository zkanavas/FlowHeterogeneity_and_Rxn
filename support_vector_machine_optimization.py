# build my SVM model
from numpy import random
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score,r2_score,explained_variance_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

randomstates = np.random.randint(1,500,100)
randomstate=1
d=2.2
heterogeneities = ['pc','Mm','EMD']
# heterogeneity = 'Mm'
kernels = ['linear','rbf','sigmoid','poly']


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
    # scores  = []
    if d == 1:
        X = np.array(z).reshape(-1,1)
    elif d == 2:
        X = np.column_stack([x,z])
    elif d == 2.1:
        X = np.column_stack([y,z])
    elif d == 2.2:
        X = np.column_stack([x,y])
    elif d == 3:
        X = np.column_stack([x,y,z])

    #scale attributes to have var=1 mean~=0
    scaler = StandardScaler()
    scaler.fit(X)
    attributes = scaler.transform(X)

    # # for randomstate in randomstates:
    # X,Y,Z,B,R = shuffle(x,y,z,behavior,rxnratio, random_state = randomstate)
    # x_train, x_test, y_train, y_test, z_train, z_test, B_train, B_test, r_train, r_test = train_test_split(X, Y, Z, B, R, train_size=0.9,random_state = randomstate)

    attributes, B, R = shuffle(attributes,behavior,rxnratio,random_state=randomstate)
    att_train,att_test,beh_train,beh_test,rxn_train,rxn_test = train_test_split(attributes,B,R,train_size=0.9,random_state=randomstate)
    #cross validation
    # kf = KFold(n_splits=len(df),shuffle=True,random_state=randomstate)
    # for train_index, test_index in kf.split(attributes):
    #     att_train, att_test = attributes[train_index],attributes[test_index]
    #     rxn_train, rxn_test = rxnratio[train_index], rxnratio[test_index]
    #     beh_train, beh_test = behavior[train_index], behavior[test_index]

    for kernel in kernels:
        if kernel =='linear':
            parameters = {'C':[1e-1,1,10,100],'tol':[1e-5,1e-3,1e-2]} 
        if kernel == 'rbf':
            parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10]}
        if kernel == 'sigmoid':
            parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10],'coef0':[0,1,10]}
        if kernel =='poly':
            parameters = {'C':[1e-1,1,10,100],'gamma':[0.0001,0.001,0.01,0.1,1,10],'degree':[2,3,4],'coef0':[0,1,10]}
        svc_ = SVC(kernel = kernel)
        clf = GridSearchCV(svc_,parameters,cv=2)

        #3-d
        clf.fit(att_train,beh_train)
        # print('3d',heterogeneity,clf.best_score_,clf.best_params_)
        dfd = pd.DataFrame.from_dict(clf.cv_results_)
        vmin = 0.25
        vmax = 1.00
        if kernel =='linear':
            fig, ax1 = plt.subplots()
            plotdf = pd.DataFrame(dfd.mean_test_score.values.reshape((4,3)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_tol))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax1,vmin=vmin,vmax=vmax)
            ax.set(xlabel='tolerance', ylabel='C',title=kernel)
            fig.suptitle(heterogeneity)
            fig.tight_layout()
            fig.savefig(heterogeneity+kernel+str(d)+'d.png')
        elif kernel == 'rbf':
            fig,ax1 = plt.subplots()
            plotdf = pd.DataFrame(dfd.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax1,vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma', ylabel='C',title=kernel)
            fig.suptitle(heterogeneity)
            fig.tight_layout()
            fig.savefig(heterogeneity+kernel+str(d)+'d.png')
        elif kernel == 'sigmoid':
            fig, (ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(12,4.8))
            plot_df0 = dfd.drop(np.where(dfd.param_coef0 != 0)[0],axis=0)
            plotdf = pd.DataFrame(plot_df0.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False, ax=ax1,vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma', ylabel='C',title=kernel +' coef = 0')

            plot_df1 = dfd.drop(np.where(dfd.param_coef0 != 1)[0],axis=0)
            plotdf = pd.DataFrame(plot_df1.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax2,vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma',title=kernel+' coef = 1')

            plot_df10 = dfd.drop(np.where(dfd.param_coef0 != 10)[0],axis=0)
            plotdf = pd.DataFrame(plot_df10.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax3,vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma',title=kernel+' coef = 10')
            fig.suptitle(heterogeneity)
            fig.tight_layout()
            fig.savefig(heterogeneity+kernel+str(d)+'d.png')
        elif kernel == 'poly':
            fig,ax1 = plt.subplots(3,3,sharey=True,figsize=(16,12),sharex=True)
            plot_df0 = dfd.drop(np.where(dfd.param_coef0 != 0)[0],axis=0)
            plot_df0.reset_index(drop=True,inplace=True)
            plot_df0_deg2 = plot_df0.drop(np.where(plot_df0.param_degree != 2)[0],axis=0)
            plotdf = pd.DataFrame(plot_df0_deg2.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[0][0],vmin=vmin,vmax=vmax)
            ax.set(ylabel='C',title=kernel+' coef = 0 degree = 2')
            
            plot_df0_deg3 = plot_df0.drop(np.where(plot_df0.param_degree != 3)[0],axis=0)
            plotdf = pd.DataFrame(plot_df0_deg3.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[0][1],vmin=vmin,vmax=vmax)
            ax.set(title=kernel+' coef = 0 degree = 3')

            plot_df0_deg4 = plot_df0.drop(np.where(plot_df0.param_degree != 4)[0],axis=0)
            plotdf = pd.DataFrame(plot_df0_deg4.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax1[0][2],vmin=vmin,vmax=vmax)
            ax.set(title=kernel +' coef = 0 degree = 4')

            plot_df1 = dfd.drop(np.where(dfd.param_coef0 != 1)[0],axis=0)
            plot_df1.reset_index(drop=True,inplace=True)
            plot_df1_deg2 = plot_df1.drop(np.where(plot_df1.param_degree != 2)[0],axis=0)
            plotdf = pd.DataFrame(plot_df1_deg2.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[1][0],vmin=vmin,vmax=vmax)
            ax.set(ylabel='C',title=kernel+' coef = 1 degree = 2')

            plot_df1_deg3 = plot_df1.drop(np.where(plot_df1.param_degree != 3)[0],axis=0)
            plotdf = pd.DataFrame(plot_df1_deg3.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[1][1],vmin=vmin,vmax=vmax)
            ax.set(title=kernel+' coef = 1 degree = 3')

            plot_df1_deg4 = plot_df1.drop(np.where(plot_df1.param_degree != 4)[0],axis=0)
            plotdf = pd.DataFrame(plot_df1_deg4.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax1[1][2],vmin=vmin,vmax=vmax)
            ax.set(title=kernel+' coef = 1 degree = 4')

            plot_df10 = dfd.drop(np.where(dfd.param_coef0 != 10)[0],axis=0)
            plot_df10.reset_index(drop=True,inplace=True)
            plot_df10_deg2 = plot_df10.drop(np.where(plot_df10.param_degree != 2)[0],axis=0)
            plotdf = pd.DataFrame(plot_df10_deg2.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[2][0],vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma', ylabel='C',title=kernel+' coef = 10 degree = 2')

            plot_df10_deg3 = plot_df10.drop(np.where(plot_df10.param_degree != 3)[0],axis=0)
            plotdf = pd.DataFrame(plot_df10_deg3.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar=False,ax=ax1[2][1],vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma',title=kernel+' coef = 10 degree = 3')

            plot_df10_deg4 = plot_df10.drop(np.where(plot_df10.param_degree != 4)[0],axis=0)
            plotdf = pd.DataFrame(plot_df10_deg4.mean_test_score.values.reshape((4,6)),index=np.unique(dfd.param_C),columns = np.unique(dfd.param_gamma))
            ax = sns.heatmap(plotdf,annot=True,cbar_kws={'label': 'Mean Test Score'},ax=ax1[2][2],vmin=vmin,vmax=vmax)
            ax.set(xlabel='gamma',title=kernel+' coef = 10 degree = 4')
            fig.suptitle(heterogeneity)
            fig.tight_layout()
            fig.savefig(heterogeneity+kernel+str(d)+'d.png')
# plt.show()
