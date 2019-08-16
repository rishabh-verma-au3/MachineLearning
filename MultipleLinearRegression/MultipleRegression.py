# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 02:06:42 2019

@author: Rishabh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 22:13:27 2019

@author: Rishabh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import dataset
dataset=pd.read_csv("50_Startups.csv")

# merics of feature and dependent variable
X=dataset.iloc[ : , :-1].values
y=dataset.iloc[ : ,4 ].values



# cATEGORICAL DATA
from sklearn.preprocessing import Imputer,OneHotEncoder,LabelEncoder
labelencoder_X=LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Avoiding the linear Dummy Trap
X=X[:,1:]
#Linear Regression model


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#appply MultipleLinear Regression due to multiple independent  variable through backward Eliminatiuon

import statsmodels.formula.api as sm  
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)#remember about values and array for this .

#optimal; x will be generated...Backward Elimination will be started
X_opt=X[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()   #fit the model for p value removex2
regressor_ols.summary()
X_opt=X[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()   #fit the model for p value remove x1
regressor_ols.summary()
X_opt=X[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()   #fit the model for p value remove x2
regressor_ols.summary()
X_opt=X[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()   #fit the model for p value remove x2
regressor_ols.summary()
X_opt=X[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()   #fit the model for p value
regressor_ols.summary()

##X1 is the point on which our model relies.

Z=dataset.iloc[ : , 0].values
y=dataset.iloc[ : ,4 ].values
Z = pd.DataFrame(Z)
from sklearn.model_selection import train_test_split
Z_train,Z_test,y_train,y_test=train_test_split(Z,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor1=LinearRegression()
regressor1.fit(Z_train,y_train)
y_pred=regressor1.predict(Z_test)

































