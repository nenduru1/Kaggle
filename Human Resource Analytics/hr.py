# -*- coding: utf-8 -*-
"""
Created on Sun May 14 11:49:15 2017

@author: Nitesh
"""

import pandas as pd
import numpy as np
import seaborn as sns
data=pd.read_csv('HR_comma_sep.csv')
data.head()

corr=data.corr()
corr=(corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
corr

from sklearn import linear_model,preprocessing,cross_validation,svm,tree,neighbors,ensemble,naive_bayes,neural_network
names=data.columns.values.tolist()
names.remove('left')
X=data[names]
y=data['left']
def values():
    for i in names:
        j= X[i].value_counts()
        return(j)
values()

def value(name):
    j= X[name].value_counts()
    return(j)
value('satisfaction_level')
value('last_evaluation')
value('number_project')
value('average_montly_hours')
value('time_spend_company')
value('Work_accident')
value('promotion_last_5years')
value('sales')
value('salary')
values(names)
enc=preprocessing.OneHotEncoder()
X=enc.fit_transform(X).toarray()
X['salary']=enc.fit_transform(X['salary']).toarray()
X.dtypes
data[names].isnull().sum()
X1=pd.get_dummies(X,columns=['sales','salary'])
scale=preprocessing.StandardScaler()
X1=scale.fit_transform(X1)
def x():    
    for i in names:
        print(i)
        
X_train,X_test,y_train,y_test=cross_validation.train_test_split(X1,y,test_size=0.3,random_state=0)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

def class_model(model):
    model.fit(X_train,y_train)
    y_pred1=model.predict(X_train)
    y_pred=model.predict(X_test)
    y_pred[y_pred <= .5]=0
    y_pred[y_pred > .5]=1
    y_pred1[y_pred1 <= .5]=0
    y_pred1[y_pred1 > .5]=1
    cm=confusion_matrix(y_test,y_pred)
    print("Training Score: ",accuracy_score(y_train,y_pred1))
    print('Testing Score: ',accuracy_score(y_test,y_pred))

model=linear_model.LinearRegression()
class_model(linear_model.LinearRegression())

class_model(linear_model.LogisticRegression())
class_model(svm.SVC(kernel='rbf'))
class_model(tree.DecisionTreeClassifier())
class_model(ensemble.RandomForestClassifier(n_estimators=100))
class_model(naive_bayes.GaussianNB())
class_model(neural_network.MLPClassifier(activation='tanh',solver='lbfgs',hidden_layer_sizes=(50,2),random_state=0))
